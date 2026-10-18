"""P15: shared patient-level 5-fold CV, fold-safe L1 selection + logistic regression.

Usage (from project root):
    python scripts/lr_cv.py
    python scripts/lr_cv.py --final-test  # only after model choices are frozen; needs metrics.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
# sklearn >= 1.8 warns that `penalty` is deprecated; penalty="l1" still applies L1 (kept to match xgboost_cv.py).
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1")

SEED = 42
TARGET = "died_in_hospital"
ID = "subject_id"
FOLDS = 5
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


def build_pipeline(x: pd.DataFrame) -> Pipeline:
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
        # L2 logistic regression; C is tuned by an inner 5-fold CV inside each training fold.
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


def main():
    project = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=project / "splits")
    p.add_argument("--output-dir", type=Path, default=project / "results" / "lr")
    p.add_argument("--final-test", action="store_true",
                   help="Fit full training sets and evaluate held-out test ONCE")
    args = p.parse_args()
    if args.final_test:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from metrics import evaluate_model, format_table  # team evaluation standard
        except ImportError:
            sys.exit("--final-test needs the team's metrics.py in the scripts/ folder")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    a_train, a_test = load_data(args.data_dir, "A")
    b_train, b_test = load_data(args.data_dir, "B")
    for part, a, b in [("train", a_train, b_train), ("test", a_test, b_test)]:
        if not a[ID].equals(b[ID]) or not a[TARGET].equals(b[TARGET]):
            raise ValueError(f"Model A/B {part} patient IDs or outcomes differ")
    folds = get_folds(a_train, args.data_dir / "cv_folds.csv")

    fold_rows, feat_rows, oof_rows, tables, summary = [], [], [], [], {}
    for name, tr, te in [("A", a_train, a_test), ("B", b_train, b_test)]:
        cols = [c for c in tr.columns if c not in DROP_FEATURES]
        x, y = tr[cols], tr[TARGET].to_numpy()
        print(f"\n{name}: train={len(tr)}, test={len(te)}, raw predictors={len(cols)}")

        oof = np.full(len(tr), np.nan)
        for fold in range(FOLDS):
            ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            pipe = build_pipeline(x).fit(x.iloc[ti], y[ti])
            oof[vi] = pipe.predict_proba(x.iloc[vi])[:, 1]
            chosen = selected_names(pipe)
            fold_rows.append({"model": name, "fold": fold, "n_train": len(ti),
                              "n_validation": len(vi), "selected_features": len(chosen),
                              "lr_C": chosen_c(pipe),
                              "validation_auroc": roc_auc_score(y[vi], oof[vi])})
            feat_rows.extend({"model": name, "fold": fold, "feature": f} for f in chosen)
            print(f"  fold {fold}: AUROC={fold_rows[-1]['validation_auroc']:.3f}, "
                  f"features={len(chosen)}, C={chosen_c(pipe):.4g}")
        oof_rows.append(pd.DataFrame({"model": name, ID: tr[ID], "fold": folds,
                                      "observed": y, "oof_probability": oof}))
        summary[name] = {"n_raw_predictors": len(cols),
                         "cv_pooled_auroc": float(roc_auc_score(y, oof))}

        if args.final_test:
            final = build_pipeline(x).fit(x, y)
            p_tr = final.predict_proba(x)[:, 1]
            p_te = final.predict_proba(te[cols])[:, 1]
            table = evaluate_model(f"LR_{name}", y, p_tr, te[TARGET].to_numpy(), p_te, oof)
            table["Cohort"] = table["Cohort"].replace({"Train (10-fold CV)": "Train (5-fold CV)"})
            tables.append(table)
            names = selected_names(final)
            pd.DataFrame({"feature": names,
                          "coefficient": final.named_steps["lr"].coef_.ravel()}) \
                .sort_values("coefficient", key=np.abs, ascending=False) \
                .to_csv(out / f"final_coefficients_{name}.csv", index=False)
            pd.DataFrame({ID: te[ID], "observed": te[TARGET],
                          "predicted_probability": p_te}) \
                .to_csv(out / f"test_predictions_{name}.csv", index=False)
            summary[name].update({"final_n_selected": len(names), "final_lr_C": chosen_c(final)})
            print(f"  final model: features={len(names)}, C={chosen_c(final):.4g}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out / "cv_fold_metrics.csv", index=False)
    feats = pd.DataFrame(feat_rows)
    feats.to_csv(out / "cv_selected_features.csv", index=False)
    feats.groupby(["model", "feature"]).size().rename("folds_selected").reset_index() \
        .sort_values(["model", "folds_selected", "feature"], ascending=[True, False, True]) \
        .to_csv(out / "cv_feature_selection_frequency.csv", index=False)
    pd.concat(oof_rows).to_csv(out / "cv_oof_predictions.csv", index=False)

    print("\nCV summary (mean / std):")
    print(folds_df.groupby("model")[["validation_auroc", "selected_features"]]
          .agg(["mean", "std"]).round(3).to_string())

    if args.final_test:
        perf = pd.concat(tables, ignore_index=True)
        perf.to_csv(out / "lr_performance.csv", index=False)
        print("\nTeam-standard performance (metrics.py):")
        print(format_table(perf).to_string(index=False))
    else:
        print("\nTest sets remain untouched. Run --final-test only after freezing model choices.")

    (out / "run_summary.json").write_text(json.dumps({
        "split_dir": str(args.data_dir), "folds": FOLDS, "seed": SEED,
        "lasso": "L1 LogisticRegression(liblinear, C=0.5), same as xgboost_cv.py",
        "lr": "LogisticRegressionCV(L2, inner 5-fold, scoring=roc_auc)",
        "C_grid": C_GRID.tolist(), "final_test_run": args.final_test, "models": summary,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
