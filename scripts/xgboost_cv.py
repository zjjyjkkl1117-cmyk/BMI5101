"""P15: shared patient-level 5-fold CV, fold-safe L1 selection + XGBoost.

Usage (from project root):
    python scripts/xgboost_cv.py
    python scripts/xgboost_cv.py --final-test  # only after model choices are frozen
"""
from pathlib import Path
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
TARGET = "died_in_hospital"
ID = "subject_id"
FOLDS = 5
# Never feed identifiers, other outcome labels, or post-outcome flags into the model.
DROP_FEATURES = {
    "subject_id", "hadm_id", "first_stay_id",
    "died_in_hospital", "died_28d", "died_in_icu", "died_or_hospice",
    "paper_would_exclude",
}
# A is intentionally a leakage-containing replication, NOT a deployable model.
# B still requires a complete timestamp/clinical feature audit before deployment.


def load_data(data_dir, name):
    tr = pd.read_csv(data_dir / f"{name}_train.csv")
    te = pd.read_csv(data_dir / f"{name}_test.csv")
    for label, df in [("train", tr), ("test", te)]:
        if df[ID].isna().any() or not df[ID].is_unique:
            raise ValueError(f"{name}_{label}: missing/duplicate subject_id")
        if not df[TARGET].isin([0, 1]).all():
            raise ValueError(f"{name}_{label}: invalid/missing target")
    if set(tr[ID]) & set(te[ID]):
        raise ValueError(f"{name}: patient overlap across train and test")
    return tr.sort_values(ID).reset_index(drop=True), te.sort_values(ID).reset_index(drop=True)


def build_pipeline(x):
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x if c not in numeric]
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True,
                                     add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocess", prep),
        ("lasso", SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                               max_iter=3000, random_state=SEED),
            threshold=1e-8,
        )),
        ("xgboost", XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=2,
            random_state=SEED,
        )),
    ])


def feature_names(pipeline):
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    mask = pipeline.named_steps["lasso"].get_support()
    return names[mask].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing A_train.csv etc.; default: project/data")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory; default: project/results")
    parser.add_argument("--final-test", action="store_true",
                        help="Fit full training sets and evaluate held-out test ONCE")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir or project / "splits"
    out = args.output_dir or project / "results"
    out.mkdir(parents=True, exist_ok=True)

    a_train, a_test = load_data(data_dir, "A")
    b_train, b_test = load_data(data_dir, "B")
    for part, a, b in [("train", a_train, b_train), ("test", a_test, b_test)]:
        if not a[ID].equals(b[ID]) or not a[TARGET].equals(b[TARGET]):
            raise ValueError(f"Model A/B {part} patient IDs or outcomes differ")
    # If an existing split manifest exists, verify it matches the uploaded A/B files.
    split_file = project / "splits" / "train_test_split.csv"
    if split_file.exists():
        manifest = pd.read_csv(split_file)
        if not manifest[ID].is_unique:
            raise ValueError("split manifest has duplicate IDs")
        assigned = dict(zip(manifest[ID], manifest["split"]))
        if any(assigned.get(v) != "train" for v in a_train[ID]) or any(
            assigned.get(v) != "test" for v in a_test[ID]
        ):
            raise ValueError("Existing split manifest does not match A/B train/test files")

    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = np.full(len(a_train), -1, dtype=int)
    for i, (_, valid_idx) in enumerate(cv.split(a_train, a_train[TARGET])):
        folds[valid_idx] = i
    fold_file = project / "splits" / "cv_folds.csv"
    if fold_file.exists():
        old = pd.read_csv(fold_file).set_index(ID)["fold"]
        new = pd.Series(folds, index=a_train[ID].to_numpy())
        if not old.sort_index().equals(new.sort_index()):
            raise ValueError("Existing cv_folds.csv differs from current folds. Do not overwrite.")
    else:
        fold_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: a_train[ID], "fold": folds}).to_csv(fold_file, index=False)

    rows = []
    selected_rows = []
    for name, tr, te in [("A", a_train, a_test), ("B", b_train, b_test)]:
        cols = [c for c in tr.columns if c not in DROP_FEATURES]
        if not cols or any(c not in te.columns for c in cols):
            raise ValueError(f"{name}: invalid feature columns")
        x, y = tr[cols], tr[TARGET]
        print(f"\n{name}: train={len(tr)}, test={len(te)}, raw predictors={len(cols)}", flush=True)
        for fold in range(FOLDS):
            ti = np.flatnonzero(folds != fold)
            vi = np.flatnonzero(folds == fold)
            pipe = build_pipeline(x)
            pipe.fit(x.iloc[ti], y.iloc[ti])
            prob = pipe.predict_proba(x.iloc[vi])[:, 1]
            chosen = feature_names(pipe)
            if not chosen:
                raise RuntimeError(f"{name} fold {fold}: LASSO selected zero features; tune C inside CV")
            rows.append({"model": name, "fold": fold, "n_train": len(ti),
                         "n_validation": len(vi), "selected_features": len(chosen),
                         "validation_auroc": roc_auc_score(y.iloc[vi], prob),
                         "validation_auprc": average_precision_score(y.iloc[vi], prob)})
            selected_rows.extend({"model": name, "fold": fold, "feature": feat}
                                 for feat in chosen)
            print(f"  fold {fold}: AUROC={rows[-1]['validation_auroc']:.3f}, "
                  f"AUPRC={rows[-1]['validation_auprc']:.3f}, features={len(chosen)}", flush=True)

        if args.final_test:
            pipe = build_pipeline(x)
            pipe.fit(x, y)
            prob = pipe.predict_proba(te[cols])[:, 1]
            final = {"model": name, "n_test": len(te),
                     "test_auroc": roc_auc_score(te[TARGET], prob),
                     "test_auprc": average_precision_score(te[TARGET], prob),
                     "n_final_features": len(feature_names(pipe))}
            (out / f"{name}_final_metrics.json").write_text(json.dumps(final, indent=2))
            pd.DataFrame({"feature": feature_names(pipe)}).to_csv(
                out / f"{name}_final_selected_features.csv", index=False)
            pd.DataFrame({ID: te[ID], "observed": te[TARGET],
                          "predicted_probability": prob}).to_csv(
                out / f"{name}_test_predictions.csv", index=False)
            joblib.dump(pipe, out / f"{name}_final_pipeline.joblib")
            print(f"  FINAL TEST {name}: AUROC={final['test_auroc']:.3f}, "
                  f"AUPRC={final['test_auprc']:.3f}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(out / "cv_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "cv_selected_features.csv", index=False)
    print("\nCV summary (mean / std):")
    print(result.groupby("model")[["validation_auroc", "validation_auprc", "selected_features"]]
          .agg(["mean", "std"]).round(3).to_string())
    print(f"\nSaved to {out}")
    if not args.final_test:
        print("Test sets remain untouched. Run --final-test only after freezing model choices.")


if __name__ == "__main__":
    main()
