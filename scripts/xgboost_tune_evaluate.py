"""P15: fold-safe XGBoost tuning and team-standard final evaluation.

Place this file AND the team's unchanged metrics.py in project/scripts/.
Run from project root:
  python scripts/xgboost_tune_evaluate.py --stage tune
  python scripts/xgboost_tune_evaluate.py --stage final

Do not run --stage final until model choices have been frozen.
"""
from pathlib import Path
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, PredefinedSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from metrics import oof_proba, evaluate_model, format_table

SEED = 42
ID = 'subject_id'
TARGET = 'died_in_hospital'
DROP = {
    'subject_id', 'hadm_id', 'first_stay_id',
    'died_in_hospital', 'died_28d', 'died_in_icu', 'died_or_hospice',
    'paper_would_exclude',
}
# Small, deliberately limited candidate grid: 3 * 2 * 2 * 2 = 24 configurations
# per dataset, 120 five-fold fits each. Expand only with training CV evidence.
GRID = {
    'lasso__estimator__C': [0.03, 0.1, 0.5],
    'xgboost__max_depth': [2, 3],
    'xgboost__min_child_weight': [3, 8],
    'xgboost__n_estimators': [100, 200],
}


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    return pd.read_csv(path)


def load_data(data_dir, name, include_test=False):
    tr = load_csv(data_dir / f'{name}_train.csv').sort_values(ID).reset_index(drop=True)
    te = None
    if include_test:
        te = load_csv(data_dir / f'{name}_test.csv').sort_values(ID).reset_index(drop=True)
    for label, frame in [('train', tr), ('test', te)]:
        if frame is None:
            continue
        if ID not in frame or TARGET not in frame:
            raise ValueError(f'{name} {label}: missing ID/target')
        if frame[ID].isna().any() or not frame[ID].is_unique:
            raise ValueError(f'{name} {label}: invalid patient IDs')
        if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
            raise ValueError(f'{name} {label}: invalid target')
    if te is not None and set(tr[ID]) & set(te[ID]):
        raise ValueError(f'{name}: train/test overlap')
    return tr, te


def build_pipeline(x):
    numerical = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x.columns if c not in numerical]
    prep = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True,
                                      keep_empty_features=True)),
            ('scaler', StandardScaler()),
        ]), numerical),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0)
    return Pipeline([
        ('preprocess', prep),
        ('lasso', SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=0.1,
                               max_iter=3000, random_state=SEED), threshold=1e-8)),
        ('xgboost', XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, objective='binary:logistic', eval_metric='logloss',
            tree_method='hist', n_jobs=2, random_state=SEED)),
    ])


def selected_features(pipe):
    names = pipe.named_steps['preprocess'].get_feature_names_out()
    mask = pipe.named_steps['lasso'].get_support()
    return names[mask].tolist()


def load_folds(project, train):
    path = project / 'splits' / 'cv_folds.csv'
    if path.exists():
        info = load_csv(path)
        if not info[ID].is_unique or set(info[ID]) != set(train[ID]):
            raise ValueError('cv_folds.csv does not match training patients')
        folds = train[[ID]].merge(info, on=ID, validate='one_to_one')['fold']
        if folds.isna().any() or set(folds.astype(int)) != set(range(5)):
            raise ValueError('cv_folds.csv must contain folds 0..4')
        return folds.astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.full(len(train), -1, dtype=int)
    for i, (_, valid) in enumerate(cv.split(train, train[TARGET])):
        folds[valid] = i
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: train[ID], 'fold': folds}).to_csv(path, index=False)
    return folds


def predictors(train, test=None):
    cols = [c for c in train if c not in DROP]
    if test is not None and any(c not in test for c in cols):
        raise ValueError('Test data lacks training feature columns')
    return cols


def data_signature(frame):
    # Stable across reruns and detects accidental changes to the training CSV.
    content = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(content).hexdigest()


def tune_one(name, train, folds, output, fast=False):
    cols = predictors(train)
    x, y = train[cols], train[TARGET]
    cv = PredefinedSplit(test_fold=folds)
    grid = list(ParameterGrid(GRID))
    if fast:
        grid = [{'lasso__estimator__C': 0.1,
                 'xgboost__max_depth': 2,
                 'xgboost__min_child_weight': 3,
                 'xgboost__n_estimators': 10}]
    rows = []
    for i, params in enumerate(grid, start=1):
        aucs, aps, nfeatures = [], [], []
        for ti, vi in cv.split():
            pipe = build_pipeline(x).set_params(**params)
            pipe.fit(x.iloc[ti], y.iloc[ti])
            n_selected = len(selected_features(pipe))
            if n_selected == 0:
                raise RuntimeError(f'{name}: LASSO selected no features: {params}')
            prob = pipe.predict_proba(x.iloc[vi])[:, 1]
            aucs.append(roc_auc_score(y.iloc[vi], prob))
            aps.append(average_precision_score(y.iloc[vi], prob))
            nfeatures.append(n_selected)
        rows.append({**params, 'mean_cv_auroc': np.mean(aucs),
                     'std_cv_auroc': np.std(aucs, ddof=1),
                     'mean_cv_auprc': np.mean(aps),
                     'mean_selected_features': np.mean(nfeatures)})
        print(f'{name} {i}/{len(grid)}: AUROC={np.mean(aucs):.3f}', flush=True)
    table = pd.DataFrame(rows).sort_values(
        ['mean_cv_auroc', 'mean_cv_auprc'], ascending=False).reset_index(drop=True)
    table.to_csv(output / f'{name}_tuning_results.csv', index=False)
    best = {k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in grid[0].items()}
    for k in best:
        val = table.loc[0, k]
        best[k] = val.item() if isinstance(val, np.generic) else val
    config = {'model': name, 'best_params': best,
              'mean_cv_auroc': float(table.loc[0, 'mean_cv_auroc']),
              'train_sha256': data_signature(train),
              'folds_sha256': hashlib.sha256(folds.tobytes()).hexdigest(),
              'seed': SEED, 'selection_metric': 'mean_cv_auroc',
              'tuning_mode': 'smoke' if fast else 'full'}
    (output / f'{name}_best_config.json').write_text(
        json.dumps(config, indent=2), encoding='utf-8')
    print(f'BEST {name}: {best}, AUROC={config["mean_cv_auroc"]:.3f}')


def final_one(name, train, test, folds, output):
    config_path = output / f'{name}_best_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Run tuning first: {config_path}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('tuning_mode') != 'full':
        raise ValueError('Smoke-test configuration cannot be used for final testing')
    if config['train_sha256'] != data_signature(train):
        raise ValueError(f'{name}: training data changed after tuning')
    if config['folds_sha256'] != hashlib.sha256(folds.tobytes()).hexdigest():
        raise ValueError(f'{name}: CV folds changed after tuning')
    cols = predictors(train, test)
    x_train, y_train = train[cols], train[TARGET]
    x_test, y_test = test[cols], test[TARGET]
    pipe = build_pipeline(x_train).set_params(**config['best_params'])
    # This is a separate 10-fold OOF evaluation per the unchanged team metrics.py.
    # It refits the entire preprocessing + LASSO + XGBoost pipeline in each fold.
    p_oof = oof_proba(pipe, x_train, y_train, n_folds=10, seed=SEED)
    pipe.fit(x_train, y_train)
    p_train = pipe.predict_proba(x_train)[:, 1]
    p_test = pipe.predict_proba(x_test)[:, 1]
    table = evaluate_model(f'XGBoost_{name}', y_train, p_train, y_test, p_test, p_oof)
    table.to_csv(output / f'{name}_performance_full.csv', index=False)
    format_table(table).to_csv(output / f'{name}_performance_formatted.csv', index=False)
    pd.DataFrame({'feature': selected_features(pipe)}).to_csv(
        output / f'{name}_selected_features_final.csv', index=False)
    pd.DataFrame({ID: test[ID], 'observed': y_test,
                  'predicted_probability': p_test}).to_csv(
        output / f'{name}_test_predictions.csv', index=False)
    joblib.dump(pipe, output / f'{name}_final_pipeline.joblib')
    print(f'\n{name} FINAL PERFORMANCE\n{format_table(table).to_string(index=False)}', flush=True)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['tune', 'final'], required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke-test', action='store_true',
                        help='One tiny configuration to check environment; not final research')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir or project / 'splits'
    output = args.output_dir or project / 'results' / 'xgboost_tuned'
    output.mkdir(parents=True, exist_ok=True)
    use_test = args.stage == 'final'
    a_train, a_test = load_data(data_dir, 'A', include_test=use_test)
    b_train, b_test = load_data(data_dir, 'B', include_test=use_test)
    if not a_train[ID].equals(b_train[ID]) or not a_train[TARGET].equals(b_train[TARGET]):
        raise ValueError('A/B training patient IDs or outcomes do not match')
    if use_test and (not a_test[ID].equals(b_test[ID]) or
                     not a_test[TARGET].equals(b_test[TARGET])):
        raise ValueError('A/B testing patient IDs or outcomes do not match')
    folds = load_folds(project, a_train)
    if args.stage == 'tune':
        for name, train in [('A', a_train), ('B', b_train)]:
            tune_one(name, train, folds, output, fast=args.smoke_test)
        print('\nTuning complete. Inspect results and freeze configurations before final testing.')
    else:
        if args.smoke_test:
            raise ValueError('--smoke-test cannot be combined with --stage final')
        tables = []
        for name, train, test in [('A', a_train, a_test), ('B', b_train, b_test)]:
            tables.append(final_one(name, train, test, folds, output))
        pd.concat(tables, ignore_index=True).to_csv(
            output / 'A_B_performance_comparison.csv', index=False)
        print('\nFinal test completed. Do not use test results to re-tune.')


if __name__ == '__main__':
    main()
