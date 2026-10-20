"""P15 post-hoc evaluation, calibration, and explainability for frozen XGBoost models.

Run AFTER xgboost_tune_evaluate.py --stage final. This script never retunes XGBoost.
It fits calibration using TRAINING out-of-fold predictions only. Test data is used
only to describe the performance of the previously frozen models and calibrators.

From project root:
  python scripts/xgboost_extended_analysis.py --stage analyze
  python scripts/xgboost_extended_analysis.py --stage analyze --models B
  python scripts/xgboost_extended_analysis.py --stage analyze --skip-shap

Dependencies: pandas numpy scipy scikit-learn xgboost shap matplotlib joblib.
Keep this file and metrics.py in project/scripts/. The original tuning script is unchanged.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression

from metrics import classification_metrics, youden_threshold

SEED = 42
ID = 'subject_id'
TARGET = 'died_in_hospital'
BOOT = 2000
N_FOLDS = 10
EPS = 1e-6


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}')
    return pd.read_csv(path).sort_values(ID).reset_index(drop=True)


def check_frames(tr, te):
    for df in (tr, te):
        if not df[ID].is_unique or df[ID].isna().any():
            raise ValueError('Missing or duplicate subject_id')
        if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
            raise ValueError('Invalid outcome labels')
    if set(tr[ID]) & set(te[ID]):
        raise ValueError('Train/test patient overlap')


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return np.log(p / (1-p))


class TemperatureScaler:
    """Binary temperature scaling: sigmoid(logit(raw_probability)/T)."""
    def fit(self, probs, y):
        z = logit(probs)
        y = np.asarray(y, dtype=int)
        result = minimize_scalar(
            lambda log_t: log_loss(y, sigmoid(z / np.exp(log_t)), labels=[0, 1]),
            bounds=(np.log(0.1), np.log(20.0)), method='bounded',
            options={'xatol': 1e-5})
        if not result.success:
            raise RuntimeError('Temperature fitting failed')
        self.temperature_ = float(np.exp(result.x))
        return self

    def predict(self, probs):
        return sigmoid(logit(probs) / self.temperature_)


def calibrator(method):
    if method == 'isotonic':
        return IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    if method == 'temperature':
        return TemperatureScaler()
    raise ValueError(method)


def fit_cal(method, probs, y):
    obj = calibrator(method)
    if method == 'isotonic':
        obj.fit(probs, y)
    else:
        obj.fit(probs, y)
    return obj


def apply_cal(method, obj, probs):
    return np.asarray(obj.predict(probs), dtype=float)


def crossfit_calibration(method, p_oof, y):
    """Train-only cross-fitted calibration diagnostics; no test-label fitting."""
    y = np.asarray(y, dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 19)
    out = np.full(len(y), np.nan)
    for ti, vi in cv.split(p_oof.reshape(-1, 1), y):
        obj = fit_cal(method, p_oof[ti], y[ti])
        out[vi] = apply_cal(method, obj, p_oof[vi])
    if not np.isfinite(out).all():
        raise RuntimeError('Cross-fitted calibration probabilities are invalid')
    return np.clip(out, 0, 1)


def calibration_slope_intercept(y, p):
    """Logistic calibration intercept and slope on logit(predicted probability)."""
    z = logit(p).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
    try:
        lr.fit(z, np.asarray(y, dtype=int))
        return float(lr.intercept_[0]), float(lr.coef_[0, 0])
    except Exception:
        return float('nan'), float('nan')


def summary_row(model, variant, cohort, y, p, threshold):
    row = classification_metrics(y, p, threshold, n_boot=BOOT, seed=SEED)
    row.update({'Model': model, 'Variant': variant, 'Cohort': cohort,
                'AUPRC_AP': float(average_precision_score(y, p)),
                'LogLoss': float(log_loss(y, np.clip(p, EPS, 1-EPS), labels=[0, 1]))})
    a, b = calibration_slope_intercept(y, p)
    row.update({'CalibrationIntercept': a, 'CalibrationSlope': b})
    return row


def paired_bootstrap(y, pa, pb, n_boot=BOOT):
    """Paired percentile CI for B minus A; same patients resampled together."""
    y = np.asarray(y, dtype=int)
    pa, pb = np.asarray(pa), np.asarray(pb)
    rng = np.random.default_rng(SEED)
    deltas_auc, deltas_ap = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) != 2:
            continue
        deltas_auc.append(roc_auc_score(y[ix], pb[ix]) - roc_auc_score(y[ix], pa[ix]))
        deltas_ap.append(average_precision_score(y[ix], pb[ix]) - average_precision_score(y[ix], pa[ix]))
    return {
        'delta_auroc_B_minus_A': float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
        'delta_auroc_CI_low': float(np.percentile(deltas_auc, 2.5)),
        'delta_auroc_CI_high': float(np.percentile(deltas_auc, 97.5)),
        'delta_ap_B_minus_A': float(average_precision_score(y, pb) - average_precision_score(y, pa)),
        'delta_ap_CI_low': float(np.percentile(deltas_ap, 2.5)),
        'delta_ap_CI_high': float(np.percentile(deltas_ap, 97.5)),
        'bootstrap_resamples': len(deltas_auc),
    }


def plot_calibration(y, pred_dict, out_file, model):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Ideal')
    for label, probs in pred_dict.items():
        frac, mean = calibration_curve(y, probs, n_bins=8, strategy='quantile')
        ax.plot(mean, frac, 'o-', label=label)
    ax.set(xlabel='Predicted mortality risk', ylabel='Observed mortality fraction',
           title=f'{model}: calibration on held-out test cohort', xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)



def decision_curve(y, prediction_variants, out, model):
    """Decision curve analysis: net benefit across threshold probabilities.

    DCA describes a hypothetical decision policy; it does not establish clinical
    utility without clinically justified thresholds and prospective evaluation.
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    prevalence = float(np.mean(y))
    thresholds = np.linspace(0.02, 0.80, 79)
    rows = []
    for t in thresholds:
        weight = t / (1.0 - t)
        rows.append({'threshold': t, 'strategy': 'treat_none', 'net_benefit': 0.0})
        rows.append({'threshold': t, 'strategy': 'treat_all',
                     'net_benefit': prevalence - (1-prevalence)*weight})
        for name, prob in prediction_variants.items():
            pred = np.asarray(prob) >= t
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            rows.append({'threshold': t, 'strategy': name,
                         'net_benefit': tp/n - (fp/n)*weight})
    table = pd.DataFrame(rows)
    table.to_csv(out / f'{model}_DCA.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, group in table.groupby('strategy', sort=False):
        ax.plot(group.threshold, group.net_benefit, label=name)
    ax.set(xlabel='Risk threshold', ylabel='Net benefit',
           title=f'{model}: decision curve (exploratory)', ylim=(-0.15, 0.3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f'{model}_DCA.png', dpi=170)
    plt.close(fig)


def shap_outputs(pipe, X, out, model, sample_n=150):
    """Explain frozen XGBoost in selected, encoded feature space (log-odds)."""
    import shap
    sample = X.sample(n=min(sample_n, len(X)), random_state=SEED)
    transformed = pipe.named_steps['preprocess'].transform(sample)
    selected = pipe.named_steps['lasso'].transform(transformed)
    names = pipe.named_steps['preprocess'].get_feature_names_out()[
        pipe.named_steps['lasso'].get_support()]
    if selected.shape[1] != len(names):
        raise ValueError('SHAP feature-name mismatch')
    # Use XGBoost's native TreeSHAP contributions to avoid SHAP/XGBoost
    # base_score serialization incompatibilities (e.g. '[2.07E-1]').
    import xgboost as xgb
    booster = pipe.named_steps['xgboost'].get_booster()
    matrix = xgb.DMatrix(selected, feature_names=booster.feature_names)
    contributions = np.asarray(booster.predict(matrix, pred_contribs=True))
    if contributions.ndim == 3:
        contributions = contributions[:, :, -1]
    if contributions.shape != (selected.shape[0], selected.shape[1] + 1):
        raise ValueError(f'Unexpected native SHAP contributions shape: {contributions.shape}')
    vals = contributions[:, :-1]  # Last column is expected-value bias.
    if vals.shape != selected.shape:
        raise ValueError(f'Unexpected SHAP shape: {vals.shape}')
    imp = pd.DataFrame({'feature': names, 'mean_abs_shap_log_odds':
                        np.mean(np.abs(vals), axis=0)}).sort_values(
                            'mean_abs_shap_log_odds', ascending=False)
    imp.to_csv(out / f'{model}_shap_importance.csv', index=False)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(vals, selected, feature_names=list(names), max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(out / f'{model}_shap_beeswarm.png', dpi=170, bbox_inches='tight')
    plt.close('all')
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top.mean_abs_shap_log_odds)
    ax.set(xlabel='Mean |SHAP| (log-odds)', title=f'{model}: SHAP importance')
    fig.tight_layout()
    fig.savefig(out / f'{model}_shap_bar.png', dpi=170)
    plt.close(fig)
    return imp


def ale_1d(pipe, background, feature, bins=10):
    """First-order empirical ALE for numeric ORIGINAL input; output probability scale.

    ALE averages model prediction differences between adjacent bin endpoints
    among observed patients in that bin; then centers by the sample mean.
    """
    x = pd.to_numeric(background[feature], errors='coerce')
    valid = x.notna()
    vals = x[valid].to_numpy(dtype=float)
    if len(vals) < 30 or len(np.unique(vals)) < 5:
        return None
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)))
    if len(edges) < 4:
        return None
    data = background.loc[valid].copy()
    idx = np.clip(np.searchsorted(edges, vals, side='right') - 1, 0, len(edges)-2)
    diffs = np.zeros(len(edges)-1)
    counts = np.zeros(len(edges)-1, dtype=int)
    for j in range(len(diffs)):
        mask = idx == j
        counts[j] = int(mask.sum())
        if counts[j] == 0:
            continue
        lower = data.loc[mask].copy()
        upper = lower.copy()
        lower[feature] = edges[j]
        upper[feature] = edges[j+1]
        diffs[j] = np.mean(pipe.predict_proba(upper)[:, 1] -
                           pipe.predict_proba(lower)[:, 1])
    accumulated = np.cumsum(diffs) - diffs/2
    accumulated -= np.average(accumulated, weights=counts)
    return pd.DataFrame({'feature': feature, 'bin_left': edges[:-1],
                         'bin_right': edges[1:], 'bin_mid': (edges[:-1]+edges[1:])/2,
                         'n': counts, 'ALE_probability': accumulated})


def ale_outputs(pipe, train_x, shap_imp, out, model, top_n=3):
    # Only numeric original columns with a corresponding selected SHAP feature.
    chosen = set(pipe.named_steps['preprocess'].get_feature_names_out()[
        pipe.named_steps['lasso'].get_support()])
    candidates = []
    for feature in train_x.select_dtypes(include=np.number).columns:
        if f'num__{feature}' not in chosen:
            continue
        if train_x[feature].nunique(dropna=True) < 5:
            continue
        weight = shap_imp.loc[shap_imp.feature == f'num__{feature}', 'mean_abs_shap_log_odds']
        candidates.append((float(weight.iloc[0]) if len(weight) else 0.0, feature))
    frames = []
    for _, feat in sorted(candidates, reverse=True)[:top_n]:
        table = ale_1d(pipe, train_x, feat)
        if table is None:
            continue
        frames.append(table)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(table.bin_mid, table.ALE_probability, marker='o')
        ax.axhline(0, linestyle='--', linewidth=1, color='gray')
        ax.set(xlabel=feat, ylabel='Centered ALE (mortality probability)',
               title=f'{model}: first-order ALE for {feat}')
        fig.tight_layout()
        fig.savefig(out / f'{model}_ALE_{feat}.png', dpi=170)
        plt.close(fig)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / f'{model}_ALE_values.csv', index=False)
    return [x.feature.iloc[0] for x in frames]


# Observational, non-interventional what-if explanations. Avoid immutable variables,
# demographic attributes, medication/treatment interventions, scores, identifiers.
PHYSIOLOGY_HINTS = ('heart_rate', 'resp_rate', 'temperature', 'spo2', 'sbp', 'dbp',
                    'mbp', 'bun', 'creatinine', 'sodium', 'potassium', 'glucose',
                    'wbc', 'hemoglobin', 'platelet', 'ph', 'lactate', 'chloride',
                    'albumin', 'pt_', 'inr')


def counterfactual_whatif(pipe, train_x, test_x, out, model, threshold,
                          max_patients=3, max_features=12):
    """Single-variable, in-distribution, illustrative contrastive what-if scenarios.

    These are NOT validated clinical counterfactuals or treatment recommendations.
    Changes do not imply that clinical interventions could achieve the result.
    """
    candidate_features = [c for c in train_x.select_dtypes(include=np.number)
                          if any(h in c.lower() for h in PHYSIOLOGY_HINTS)
                          and train_x[c].nunique(dropna=True) >= 8]
    # Rank by XGBoost importance in selected encoded space.
    names = pipe.named_steps['preprocess'].get_feature_names_out()
    selected_names = names[pipe.named_steps['lasso'].get_support()]
    gains = pipe.named_steps['xgboost'].feature_importances_
    weights = dict(zip(selected_names, gains))
    candidate_features.sort(key=lambda c: weights.get(f'num__{c}', 0), reverse=True)
    candidate_features = candidate_features[:max_features]
    base_probs = pipe.predict_proba(test_x)[:, 1]
    # Show patients near the training-derived threshold (avoid patient identifiers).
    order = np.argsort(np.abs(base_probs - threshold))[:max_patients]
    rows = []
    for rank, pos in enumerate(order, 1):
        base = test_x.iloc[[pos]].copy()
        base_prob = float(base_probs[pos])
        desired = 0 if base_prob >= threshold else 1
        proposals = []
        for feat in candidate_features:
            observed = train_x[feat].dropna().astype(float)
            if len(observed) < 20:
                continue
            # Within training distribution, not arbitrary extrapolation.
            for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                value = float(observed.quantile(q))
                if pd.notna(base.iloc[0][feat]) and np.isclose(value, base.iloc[0][feat]):
                    continue
                modified = base.copy()
                modified[feat] = value
                pred = float(pipe.predict_proba(modified)[0, 1])
                flipped = (pred >= threshold) == bool(desired)
                proposals.append((flipped, abs(pred - base_prob), feat, value, pred))
        if not proposals:
            continue
        flips = [p for p in proposals if p[0]]
        if flips:
            best = min(flips, key=lambda t: t[1])
        else:
            best = max(proposals, key=lambda t: t[1])
        rows.append({'example_number': rank, 'baseline_probability': base_prob,
                     'threshold_train_only': threshold, 'changed_feature': best[2],
                     'original_value': base.iloc[0][best[2]], 'hypothetical_value': best[3],
                     'hypothetical_probability': best[4], 'crossed_threshold': bool(best[0]),
                     'interpretation': 'Illustrative association only; not a clinical intervention'})
    result = pd.DataFrame(rows)
    result.to_csv(out / f'{model}_counterfactual_whatif.csv', index=False)
    return result


def run_model(name, project, result_dir, out, skip_shap=False, skip_ale=False,
              skip_counterfactual=False):
    train = read_csv(project / 'splits' / f'{name}_train.csv')
    test = read_csv(project / 'splits' / f'{name}_test.csv')
    check_frames(train, test)
    pipe_path = result_dir / f'{name}_final_pipeline.joblib'
    if not pipe_path.exists():
        raise FileNotFoundError(f'Missing frozen model {pipe_path}. Run --stage final first.')
    pipe = joblib.load(pipe_path)  # Load only trusted files you generated locally.
    features = list(pipe.feature_names_in_)
    if any(c not in train or c not in test for c in features):
        raise ValueError(f'{name}: frozen model input columns not found in CSV')
    x_train, x_test = train[features], test[features]
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    print(f'{name}: computing training 10-fold OOF probabilities...', flush=True)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    p_oof = cross_val_predict(clone(pipe), x_train, y_train, cv=cv,
                               method='predict_proba')[:, 1]
    p_train = pipe.predict_proba(x_train)[:, 1]
    p_test = pipe.predict_proba(x_test)[:, 1]
    variants_train = {'raw': p_oof}
    variants_test = {'raw': p_test}
    fitted = {}
    for method in ('temperature', 'isotonic'):
        print(f'{name}: training-only calibration: {method}', flush=True)
        variants_train[method] = crossfit_calibration(method, p_oof, y_train)
        obj = fit_cal(method, p_oof, y_train)
        fitted[method] = obj
        variants_test[method] = np.clip(apply_cal(method, obj, p_test), 0, 1)
        joblib.dump(obj, out / f'{name}_{method}_calibrator.joblib')
    # Selection is exclusively by cross-fitted TRAIN Brier score, not test performance.
    train_briers = {key: float(brier_score_loss(y_train, value))
                    for key, value in variants_train.items()}
    selected = min(train_briers, key=train_briers.get)
    (out / f'{name}_calibration_selection.json').write_text(
        json.dumps({'selection_data': 'train only (cross-fitted calibration)',
                    'selection_metric': 'Brier', 'selected_method': selected,
                    'train_brier_by_method': train_briers,
                    'note': 'Selection after initial test inspection is exploratory.'}, indent=2))
    rows = []
    for method in ('raw', 'temperature', 'isotonic'):
        threshold = youden_threshold(y_train, variants_train[method])
        rows.append(summary_row(name, method, 'Train (10-fold OOF; calibration crossfit)',
                                y_train, variants_train[method], threshold))
        rows.append(summary_row(name, method, 'Train (apparent)', y_train,
                                p_train if method == 'raw' else
                                apply_cal(method, fitted[method], p_train), threshold))
        rows.append(summary_row(name, method, 'Test', y_test,
                                variants_test[method], threshold))
    table = pd.DataFrame(rows)
    table.to_csv(out / f'{name}_extended_metrics.csv', index=False)
    plot_calibration(y_test, variants_test, out / f'{name}_calibration.png', name)
    decision_curve(y_test, variants_test, out, name)
    pd.DataFrame({'observed': y_test, **{f'prob_{k}': v for k, v in variants_test.items()}}).to_csv(
        out / f'{name}_test_probabilities_NO_ID.csv', index=False)
    if not skip_shap:
        print(f'{name}: SHAP...', flush=True)
        importance = shap_outputs(pipe, x_test, out, name)
        if not skip_ale:
            print(f'{name}: ALE...', flush=True)
            ale_outputs(pipe, x_train, importance, out, name)
    elif not skip_ale:
        warnings.warn('ALE skipped because --skip-shap was selected')
    if not skip_counterfactual:
        print(f'{name}: illustrative counterfactual what-if scenarios...', flush=True)
        # Counterfactuals explain the ORIGINAL XGBoost prediction, so use raw threshold.
        threshold_raw = youden_threshold(y_train, p_oof)
        counterfactual_whatif(pipe, x_train, x_test, out, name, threshold_raw)
    print(f'{name}: selected calibration = {selected}; training crossfit Brier = {train_briers}', flush=True)
    return {'test_y': y_test, 'test_ids': test[ID].to_numpy(),
            'test_prob_raw': p_test, 'test_prob_selected': variants_test[selected],
            'selected': selected, 'metrics': table}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['analyze'], required=True)
    parser.add_argument('--models', nargs='+', choices=['A', 'B'], default=['A', 'B'])
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--skip-shap', action='store_true')
    parser.add_argument('--skip-ale', action='store_true')
    parser.add_argument('--skip-counterfactual', action='store_true')
    args = parser.parse_args()
    project = args.project_dir.resolve()
    result_dir = args.results_dir or project / 'results' / 'xgboost_tuned'
    out = args.output_dir or result_dir / 'extended_analysis'
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.models:
        results[name] = run_model(name, project, result_dir, out, args.skip_shap,
                                  args.skip_ale, args.skip_counterfactual)
    pd.concat([results[n]['metrics'] for n in args.models], ignore_index=True).to_csv(
        out / 'A_B_extended_metrics.csv', index=False)
    if 'A' in results and 'B' in results:
        a, b = results['A'], results['B']
        if not np.array_equal(a['test_ids'], b['test_ids']) or not np.array_equal(a['test_y'], b['test_y']):
            raise ValueError('A/B test patients/outcomes do not match for paired bootstrap')
        comparison = {
            'raw': paired_bootstrap(a['test_y'], a['test_prob_raw'], b['test_prob_raw']),
            'train_selected_calibration': paired_bootstrap(
                a['test_y'], a['test_prob_selected'], b['test_prob_selected']),
            'calibration_A': a['selected'], 'calibration_B': b['selected'],
        }
        (out / 'A_B_paired_bootstrap.json').write_text(json.dumps(comparison, indent=2))
    print(f'Finished. Output: {out}', flush=True)
    print('CAUTION: models already tested previously; these are exploratory post-hoc analyses, not new independent validation.')


if __name__ == '__main__':
    main()
