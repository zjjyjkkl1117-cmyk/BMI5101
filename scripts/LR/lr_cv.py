"""P15: fold-safe L1 selection + logistic regression, tuned and evaluated like XGBoost.

Same protocol as xgboost_tune_evaluate.py:
  --stage tune   LASSO C in {0.03, 0.1, 0.5}, chosen by mean AUROC over the shared
                 5-fold splits/cv_folds.csv (test set untouched)
  --stage final  refit with the frozen C; the CV row and Youden threshold come from
                 metrics.py's 10-fold out-of-fold predictions; test set evaluated once
LR's own L2 strength is tuned by an inner 5-fold CV inside every training fit.
Numeric predictors are clipped to the 1st-99th percentile of the training data before
imputation (LR is sensitive to extreme values; tree models are not).

Usage (from project root; metrics.py must be in scripts/; run from project root):
    python scripts/LR/lr_cv.py --stage tune
    python scripts/LR/lr_cv.py --stage final
    python scripts/LR/lr_cv.py --stage tune --no-clip   # sensitivity analysis without clipping
    python scripts/LR/lr_cv.py --stage final --no-clip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

warnings.filterwarnings("ignore", category=FutureWarning)
# sklearn >= 1.8 warns that `penalty` is deprecated; penalty="l1" still applies L1 (kept to match xgboost_cv.py).
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1")

SEED = 42
TARGET = "died_in_hospital"
ID = "subject_id"
FOLDS = 5
LASSO_C_GRID = [0.03, 0.1, 0.5]  # same candidates as xgboost_tune_evaluate.py
C_GRID = np.logspace(-3, 2, 11)  # candidate C values for LR: 0.001 ... 100
# Same as xgboost_cv.py: never feed identifiers, other outcome labels, or post-outcome flags.
DROP_FEATURES = {
    "subject_id", "hadm_id", "first_stay_id",
    "died_in_hospital", "died_28d", "died_in_icu", "died_or_hospice",
    "paper_would_exclude",
}


def load_data(data_dir: Path, name: str):
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


def get_folds(a_train: pd.DataFrame, fold_file: Path) -> np.ndarray:
    """Same fold assignment as xgboost_cv.py; an existing cv_folds.csv must match."""
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = np.full(len(a_train), -1, dtype=int)
    for i, (_, valid_idx) in enumerate(cv.split(a_train, a_train[TARGET])):
        folds[valid_idx] = i
    new = pd.Series(folds, index=a_train[ID].to_numpy())
    if fold_file.exists():
        old = pd.read_csv(fold_file).set_index(ID)["fold"]
        if not old.sort_index().equals(new.sort_index()):
            raise ValueError("Existing cv_folds.csv differs from current folds. Do not overwrite.")
        print(f"Using existing {fold_file.name} (matches xgboost_cv.py)")
        return old.reindex(a_train[ID]).to_numpy()
    pd.DataFrame({ID: a_train[ID], "fold": folds}).to_csv(fold_file, index=False)
    print(f"Saved fold assignment to {fold_file}")
    return folds


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to [lower, upper] quantiles learned on the training data.

    Binary columns (two or fewer distinct values) and all-missing columns are left
    unchanged; missing values stay missing.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=float)
        self.low_ = np.full(X.shape[1], -np.inf)
        self.high_ = np.full(X.shape[1], np.inf)
        for j in range(X.shape[1]):
            col = X[~np.isnan(X[:, j]), j]
            if np.unique(col).size > 2:
                self.low_[j], self.high_[j] = np.quantile(col, [self.lower, self.upper])
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=float, reset=False)
        return np.clip(X, self.low_, self.high_)


def build_pipeline(x: pd.DataFrame, clip: bool = True, lasso_c: float = 0.1) -> Pipeline:
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x if c not in numeric]
    numeric_steps = [("clip", QuantileClipper())] if clip else []
    numeric_steps += [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True,
                                 add_indicator=True)),
        ("scale", StandardScaler()),
    ]
    prep = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocess", prep),
        ("lasso", SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear", C=lasso_c,
                               max_iter=3000, random_state=SEED),
            threshold=1e-8,
        )),
        # L2 logistic regression; C is tuned by an inner 5-fold CV inside each training fit.
        ("lr", LogisticRegressionCV(
            Cs=C_GRID, cv=StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED),
            scoring="roc_auc", max_iter=5000, use_legacy_attributes=False,
        )),
    ])


def selected_names(pipe: Pipeline) -> np.ndarray:
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    return names[pipe.named_steps["lasso"].get_support()]


def chosen_c(pipe: Pipeline) -> float:
    return float(np.ravel(pipe.named_steps["lr"].C_)[0])


def predictors(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_FEATURES]


def data_signature(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def output_dir(project: Path, clip: bool) -> Path:
    return project / "results" / ("lr" if clip else "lr_no_clip")


def load_best_config(out: Path, name: str) -> dict:
    path = out / f"{name}_best_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Run --stage tune first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def tune_one(name, tr, folds, out, clip):
    cols = predictors(tr)
    x, y = tr[cols], tr[TARGET].to_numpy()
    rows, fold_rows, feat_rows = [], [], []
    for lasso_c in LASSO_C_GRID:
        aucs, aps, nfeat = [], [], []
        for fold in range(FOLDS):
            ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            pipe = build_pipeline(x, clip, lasso_c).fit(x.iloc[ti], y[ti])
            chosen = selected_names(pipe)
            if len(chosen) == 0:
                raise RuntimeError(f"{name}: LASSO selected no features at C={lasso_c}")
            prob = pipe.predict_proba(x.iloc[vi])[:, 1]
            aucs.append(roc_auc_score(y[vi], prob))
            aps.append(average_precision_score(y[vi], prob))
            nfeat.append(len(chosen))
            fold_rows.append({"model": name, "lasso_C": lasso_c, "fold": fold,
                              "selected_features": len(chosen), "lr_C": chosen_c(pipe),
                              "validation_auroc": aucs[-1], "validation_auprc": aps[-1]})
            feat_rows.extend({"model": name, "lasso_C": lasso_c, "fold": fold, "feature": f}
                             for f in chosen)
        rows.append({"lasso_C": lasso_c, "mean_cv_auroc": np.mean(aucs),
                     "std_cv_auroc": np.std(aucs, ddof=1), "mean_cv_auprc": np.mean(aps),
                     "mean_selected_features": np.mean(nfeat)})
        print(f"  {name} LASSO C={lasso_c}: AUROC={np.mean(aucs):.3f} "
              f"(sd {np.std(aucs, ddof=1):.3f}), features={np.mean(nfeat):.0f}", flush=True)
    table = pd.DataFrame(rows).sort_values(["mean_cv_auroc", "mean_cv_auprc"], ascending=False)
    table.to_csv(out / f"{name}_tuning_results.csv", index=False)
    best = table.iloc[0]
    config = {"model": name, "lasso_C": float(best["lasso_C"]),
              "mean_cv_auroc": float(best["mean_cv_auroc"]), "clip": clip,
              "train_sha256": data_signature(tr),
              "folds_sha256": hashlib.sha256(folds.tobytes()).hexdigest(),
              "seed": SEED, "selection_metric": "mean_cv_auroc"}
    (out / f"{name}_best_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"BEST {name}: LASSO C={config['lasso_C']}, AUROC={config['mean_cv_auroc']:.3f}")
    return pd.DataFrame(fold_rows), pd.DataFrame(feat_rows)


def final_one(name, tr, te, folds, out, clip):
    from metrics import evaluate_model, oof_proba  # team evaluation standard
    config = load_best_config(out, name)
    if config["train_sha256"] != data_signature(tr):
        raise ValueError(f"{name}: training data changed after tuning")
    if config["folds_sha256"] != hashlib.sha256(folds.tobytes()).hexdigest():
        raise ValueError(f"{name}: CV folds changed after tuning")
    if config["clip"] != clip:
        raise ValueError(f"{name}: tuned with clip={config['clip']}, final run uses clip={clip}")
    cols = predictors(tr)
    x, y = tr[cols], tr[TARGET].to_numpy()
    pipe = build_pipeline(x, clip, config["lasso_C"])
    # Same as xgboost_tune_evaluate.py: 10-fold OOF from the unchanged metrics.py.
    p_oof = oof_proba(pipe, x, y, n_folds=10, seed=SEED)
    pipe.fit(x, y)
    p_train = pipe.predict_proba(x)[:, 1]
    p_test = pipe.predict_proba(te[cols])[:, 1]
    table = evaluate_model(f"LR_{name}", y, p_train, te[TARGET].to_numpy(), p_test, p_oof)
    names = selected_names(pipe)
    pd.DataFrame({"feature": names, "coefficient": pipe.named_steps["lr"].coef_.ravel()}) \
        .sort_values("coefficient", key=np.abs, ascending=False) \
        .to_csv(out / f"final_coefficients_{name}.csv", index=False)
    pd.DataFrame({ID: te[ID], "observed": te[TARGET], "predicted_probability": p_test}) \
        .to_csv(out / f"test_predictions_{name}.csv", index=False)
    pd.DataFrame({ID: tr[ID], "observed": y, "oof_probability_10fold": p_oof,
                  "apparent_probability": p_train}) \
        .to_csv(out / f"train_predictions_{name}.csv", index=False)
    print(f"  {name}: LASSO C={config['lasso_C']}, features={len(names)}, LR C={chosen_c(pipe):.4g}")
    return table, {"lasso_C": config["lasso_C"], "final_n_selected": len(names),
                   "final_lr_C": chosen_c(pipe)}


def main():
    project = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["tune", "final"], required=True)
    p.add_argument("--data-dir", type=Path, default=project / "splits")
    p.add_argument("--no-clip", action="store_true",
                   help="Sensitivity analysis: skip the 1st-99th percentile clipping")
    args = p.parse_args()
    clip = not args.no_clip
    out = output_dir(project, clip)
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(Path(__file__).resolve().parent),  # scripts/LR (lr_cv.py)
                    str(Path(__file__).resolve().parents[1])]  # scripts (metrics.py)

    a_train, a_test = load_data(args.data_dir, "A")
    b_train, b_test = load_data(args.data_dir, "B")
    for part, a, b in [("train", a_train, b_train), ("test", a_test, b_test)]:
        if not a[ID].equals(b[ID]) or not a[TARGET].equals(b[TARGET]):
            raise ValueError(f"Model A/B {part} patient IDs or outcomes differ")
    folds = get_folds(a_train, args.data_dir / "cv_folds.csv")

    if args.stage == "tune":
        fold_tables, feat_tables = [], []
        for name, tr in [("A", a_train), ("B", b_train)]:
            print(f"\n{name}: train={len(tr)}, raw predictors={len(predictors(tr))}")
            f1, f2 = tune_one(name, tr, folds, out, clip)
            fold_tables.append(f1)
            feat_tables.append(f2)
        pd.concat(fold_tables).to_csv(out / "cv_fold_metrics.csv", index=False)
        feats = pd.concat(feat_tables)
        feats.groupby(["model", "lasso_C", "feature"]).size().rename("folds_selected") \
            .reset_index().sort_values(["model", "lasso_C", "folds_selected", "feature"],
                                       ascending=[True, True, False, True]) \
            .to_csv(out / "cv_feature_selection_frequency.csv", index=False)
        print("\nTuning complete. Test sets untouched. Freeze the choice, then run --stage final.")
        return

    from metrics import format_table
    tables, summary = [], {}
    for name, tr, te in [("A", a_train, a_test), ("B", b_train, b_test)]:
        table, info = final_one(name, tr, te, folds, out, clip)
        tables.append(table)
        summary[name] = info
    perf = pd.concat(tables, ignore_index=True)
    perf.to_csv(out / "lr_performance.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps({
        "split_dir": str(args.data_dir), "seed": SEED, "clip_numeric_1_99": clip,
        "lasso_C_grid": LASSO_C_GRID, "lr_C_grid": C_GRID.tolist(),
        "cv_row": "10-fold OOF from metrics.py (same as xgboost_tune_evaluate.py)",
        "models": summary}, indent=2), encoding="utf-8")
    print("\nTeam-standard performance (metrics.py):")
    print(format_table(perf).to_string(index=False))
    print("\nFinal test completed. Do not use test results to re-tune.")


if __name__ == "__main__":
    main()
