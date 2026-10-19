"""P15: Gradient boosting (GBM) with fold-safe LASSO selection and nested patient-level CV.

Variants : A (replication, contains leakage), B (leakage-reduced),
           B_noscores (B without SAPS II / APS III; sensitivity analysis)
Arms     : lasso (GBM on LASSO-selected features), all (GBM on all eligible features)

Usage (from the project root):
    python scripts/gbm_cv.py               # stage 1: data audit, nested CV, OOF results,
                                           #          threshold, final refit on training set.
                                           #          Test-set features are never read.
    python scripts/gbm_cv.py --final-test  # stage 2: evaluate the frozen models ONCE on test.

Inputs (written by scripts/create_train_test_split.py), in --data-dir (default ./splits):
    A_train.csv, A_test.csv, B_train.csv, B_test.csv, train_test_split.csv,
    cv_folds.csv (optional: the shared folds from xgboost_cv.py; created if absent)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
# parallel workers must be able to import gbm_pipeline too
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(SCRIPTS)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import PredefinedSplit, StratifiedKFold  # noqa: E402

import gbm_evaluate as ge  # noqa: E402
import gbm_pipeline as gp  # noqa: E402

SEED, ID, TARGET = gp.SEED, gp.ID, gp.TARGET
OUTER_FOLDS = 5
RECAL_SLOPE_RANGE = (0.8, 1.2)     # prespecified: recalibrate only outside this range
COMPARISONS = [  # (variant_1, arm_1, variant_2, arm_2): delta AUC = first - second
    ("A", "lasso", "A", "all"), ("B", "lasso", "B", "all"),
    ("B_noscores", "lasso", "B_noscores", "all"),
    ("A", "lasso", "B", "lasso"), ("A", "all", "B", "all"),
    ("B", "lasso", "B_noscores", "lasso"), ("B", "all", "B_noscores", "all"),
]


class QA:
    def __init__(self):
        self.lines = []

    def check(self, ok, msg):
        self.lines.append(("PASS" if ok else "FAIL") + "  " + msg)
        print(("  [PASS] " if ok else "  [FAIL] ") + msg, flush=True)
        if not ok:
            raise AssertionError(msg)

    def note(self, msg):
        self.lines.append("NOTE  " + msg)
        print("  [NOTE] " + msg, flush=True)

    def write(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt_ci(est, lo, hi, d=3):
    return f"{est:.{d}f} ({lo:.{d}f}-{hi:.{d}f})"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def read_frame(path, usecols=None):
    df = pd.read_csv(path, usecols=usecols)
    if df[ID].isna().any() or not df[ID].is_unique:
        raise ValueError(f"{path.name}: missing or duplicate subject_id")
    if TARGET in df and not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{path.name}: invalid or missing {TARGET}")
    for c in df.columns[df.dtypes == bool]:      # True/False flags -> 0/1, kept numeric
        df[c] = df[c].astype(int)
    return df.sort_values(ID).reset_index(drop=True)


def load_training(data_dir, qa):
    a_tr = read_frame(data_dir / "A_train.csv")
    b_tr = read_frame(data_dir / "B_train.csv")
    # test: IDs only in stage 1 - no features, no outcomes
    a_te_ids = read_frame(data_dir / "A_test.csv", usecols=[ID])[ID]
    b_te_ids = read_frame(data_dir / "B_test.csv", usecols=[ID])[ID]
    qa.check(a_tr[ID].equals(b_tr[ID]) and a_tr[TARGET].equals(b_tr[TARGET]),
             "Model A and B training files contain the same patients and outcomes")
    qa.check(a_te_ids.equals(b_te_ids), "Model A and B test files contain the same patients")
    qa.check(not set(a_tr[ID]) & set(a_te_ids),
             f"No patient in both training ({len(a_tr)}) and test ({len(a_te_ids)})")
    manifest_path = data_dir / "train_test_split.csv"
    if manifest_path.exists():
        m = pd.read_csv(manifest_path)
        assigned = dict(zip(m[ID], m["split"]))
        ok = (m[ID].is_unique
              and all(assigned.get(v) == "train" for v in a_tr[ID])
              and all(assigned.get(v) == "test" for v in a_te_ids)
              and len(m) == len(a_tr) + len(a_te_ids))
        qa.check(ok, "Train/test files match train_test_split.csv (seed 42, 70/30, stratified)")
    else:
        qa.note("train_test_split.csv not found; split taken from the A/B files as given")
    return {"A": a_tr, "B": b_tr}, a_te_ids


def get_folds(data_dir, a_tr, qa):
    """The shared outer folds: identical recipe to xgboost_cv.py."""
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    recipe = np.full(len(a_tr), -1, dtype=int)
    for i, (_, v) in enumerate(cv.split(a_tr, a_tr[TARGET])):
        recipe[v] = i
    path = data_dir / "cv_folds.csv"
    if path.exists():
        f = pd.read_csv(path).set_index(ID)["fold"]
        qa.check(set(f.index) == set(a_tr[ID]) and f.index.is_unique,
                 "cv_folds.csv covers exactly the training patients, once each")
        folds = f.reindex(a_tr[ID]).to_numpy().astype(int)
        qa.note("Using existing cv_folds.csv (shared with XGBoost); "
                + ("identical to the xgboost_cv.py recipe" if np.array_equal(folds, recipe)
                   else "it differs from the xgboost_cv.py recipe"))
    else:
        folds = recipe
        pd.DataFrame({ID: a_tr[ID], "fold": folds}).to_csv(path, index=False)
        qa.note("cv_folds.csv created with the xgboost_cv.py recipe (StratifiedKFold, seed 42)")
    qa.check(sorted(np.unique(folds)) == list(range(OUTER_FOLDS)),
             f"{OUTER_FOLDS} outer folds, every training patient in exactly one")
    y = a_tr[TARGET].to_numpy()
    qa.check(a_tr[ID].is_unique,
             "One row per patient, so folds are patient-grouped (stratified group CV)")
    prev = [y[folds == k].mean() for k in range(OUTER_FOLDS)]
    qa.note("Fold mortality: " + ", ".join(f"{p:.3f}" for p in prev))
    return folds


def audit(train, variants, out, qa):
    rows = []
    for v in variants:
        spec = gp.VARIANTS[v]
        df = train[spec["data"]]
        missing = [f for f in spec["features"] if f not in df.columns]
        qa.check(not missing, f"{v}: all {len(spec['features'])} candidate features present"
                 + (f" (missing: {missing})" if missing else ""))
        bad = [f for f in spec["features"] if gp.FORBIDDEN.search(f)]
        qa.check(not bad, f"{v}: no identifier/outcome/DNR/CMO/discharge column among predictors")
        if spec["data"] == "B":
            leak = set(spec["features"]) & set(gp.LEAKY_MODEL_A_ONLY)
            qa.check(not leak, f"{v}: none of the Model A leakage variables")
        for f in spec["features"]:
            s = df[f]
            num = pd.api.types.is_numeric_dtype(s)
            rows.append({
                "variant": v, "feature": f, "dtype": str(s.dtype),
                "n_unique": int(s.nunique()),
                "train_missing_pct": round(100 * s.isna().mean(), 2),
                "min": s.min() if num else None,
                "median": s.median() if num else None,
                "max": s.max() if num else None,
                "n_negative": int((s < 0).sum()) if num else None,
                "structural_zero": gp.STRUCTURAL_ZERO.get(f, ""),
                "timing_and_leakage": gp.TIMING_NOTES.get(f, ""),
            })
    aud = pd.DataFrame(rows)
    aud.to_csv(out / "00_data_audit.csv", index=False)
    # every column NOT used, with the reason - nothing is dropped silently
    used = {f for v in variants for f in gp.VARIANTS[v]["features"]}
    excl = []
    for name, df in train.items():
        for c in df.columns:
            if c in used:
                continue
            if c in (ID, "hadm_id", "first_stay_id"):
                why = "identifier"
            elif c.startswith("died_") or c == "hospital_expire_flag":
                why = "outcome label"
            elif gp.FORBIDDEN.search(c):
                why = "excluded by design (DNR/CMO, ICU stay length, discharge info, flags)"
            elif c in gp.LEAKY_MODEL_A_ONLY:
                why = "Model A leakage variable (not allowed in Model B)"
            else:
                why = "not in the paper-matched candidate set"
            excl.append({"file": name, "column": c, "reason": why,
                         "train_missing_pct": round(100 * df[c].isna().mean(), 2)})
    pd.DataFrame(excl).to_csv(out / "00_excluded_columns.csv", index=False)
    y = train["A"][TARGET]
    qa.note(f"Training set: {len(y)} patients, {int(y.sum())} deaths ({y.mean():.1%})")
    return aud


# --------------------------------------------------------------------------- #
# QA helpers
# --------------------------------------------------------------------------- #
def check_fold_fit(pipe, x_tr, y_tr, arm):
    """True if imputation/scaling/LASSO match a fit on exactly these training rows."""
    z = pipe.named_steps["zeros"].transform(x_tr)
    prep = pipe.named_steps["preprocess"]
    num_cols = prep.transformers_[0][2]
    num = prep.named_transformers_["numeric"]
    expected = z[num_cols].median(skipna=True).to_numpy(dtype=float)
    expected = np.where(np.isnan(expected), 0.0, expected)
    ok = np.allclose(num.named_steps["impute"].statistics_, expected, equal_nan=True)
    xt = prep.transform(z)
    # a scaler fitted on these rows leaves every scaled column with mean 0 on them
    k = len(num.named_steps["scale"].mean_)
    ok &= np.allclose(xt[:, :k].mean(axis=0), 0.0, atol=1e-8)
    if arm == "lasso":
        lasso = pipe.named_steps["lasso"].estimator_
        refit = gp.l1_logistic(lasso.C).fit(xt, y_tr)
        ok &= np.allclose(refit.coef_, lasso.coef_, atol=1e-8)
    return bool(ok)


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
def stage1(args, out, qa):
    train, test_ids = load_training(args.data_dir, qa)
    a_tr = train["A"]
    folds = get_folds(args.data_dir, a_tr, qa)
    audit(train, args.variants, out, qa)
    ids = a_tr[ID].to_numpy()
    y = a_tr[TARGET].to_numpy().astype(int)
    inner = StratifiedKFold(n_splits=gp.INNER_FOLDS, shuffle=True, random_state=SEED)
    final_cv = PredefinedSplit(folds)
    for sub in ("cv", "models", "features", "tables", "figures"):
        (out / sub).mkdir(exist_ok=True)

    fold_rows, sel_rows, oof_rows, sel_tables, frozen = [], [], [], [], {}
    oof = {}
    for v in args.variants:
        spec = gp.VARIANTS[v]
        x = train[spec["data"]][spec["features"]]
        for arm in gp.ARMS:
            key = f"{v}_{arm}"
            print(f"\n=== {spec['label']} | {gp.ARMS[arm]} | "
                  f"{x.shape[1]} candidates ===", flush=True)
            p_oof = np.full(len(y), np.nan)
            fold_fit_ok = True
            for k in range(OUTER_FOLDS):
                t0 = time.time()
                tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
                assert not set(tr) & set(va)
                pipe, info = gp.fit_tuned(x.iloc[tr], y[tr], arm, inner,
                                          n_jobs=args.n_jobs, n_iter=args.n_iter,
                                          Cs=args.lasso_cs)
                fold_fit_ok &= check_fold_fit(pipe, x.iloc[tr], y[tr], arm)
                p = pipe.predict_proba(x.iloc[va])[:, 1]
                p_oof[va] = p
                fold_rows.append({
                    "variant": v, "arm": arm, "fold": k, "n_train": len(tr),
                    "n_validation": len(va), "validation_deaths": int(y[va].sum()),
                    "validation_auc": roc_auc_score(y[va], p),
                    "validation_log_loss": log_loss(y[va], p),
                    "validation_brier": float(np.mean((p - y[va]) ** 2)),
                    "inner_cv_log_loss": info["inner_cv_log_loss"],
                    "lasso_C": info.get("lasso_C"), "n_features": info["n_features"],
                    **{f"gbm_{kk}": vv for kk, vv in info["gbm_params"].items()},
                    "seconds": round(time.time() - t0, 1)})
                if arm == "lasso":
                    sel_rows += [{"variant": v, "fold": k, "feature": f}
                                 for f in info["features"]]
                print(f"  fold {k}: AUC {fold_rows[-1]['validation_auc']:.3f}  "
                      f"features {info['n_features']}  C {info.get('lasso_C', '-')}  "
                      f"{fold_rows[-1]['seconds']}s", flush=True)
            qa.check(not np.isnan(p_oof).any(),
                     f"{key}: every training patient has exactly one out-of-fold prediction")
            qa.check(fold_fit_ok, f"{key}: imputer medians, scaler"
                     + (" and LASSO coefficients" if arm == "lasso" else "")
                     + " in every outer fold equal a fit on that fold's training rows only")
            oof[key] = p_oof
            oof_rows.append(pd.DataFrame({"variant": v, "arm": arm, ID: ids, "fold": folds,
                                          "observed": y, "predicted": p_oof}))

            # final model: same tuning on all training patients (shared folds as CV)
            t0 = time.time()
            final, finfo = gp.fit_tuned(x, y, arm, final_cv, n_jobs=args.n_jobs,
                                        n_iter=args.n_iter, Cs=args.lasso_cs)
            path = out / "models" / f"{key}_final_pipeline.joblib"
            joblib.dump(final, path)
            thr = ge.youden_threshold(y, p_oof)
            cal = ge.calibration_stats(y, p_oof)
            recal = not (RECAL_SLOPE_RANGE[0] <= cal["cal_slope"] <= RECAL_SLOPE_RANGE[1])
            frozen[key] = {
                "variant": v, "arm": arm, "features_candidates": spec["features"],
                "threshold_youden_oof": thr,
                "recalibration": {"applied": recal, "a": cal["recal_a"], "b": cal["recal_b"],
                                  "rule": f"OOF calibration slope outside {RECAL_SLOPE_RANGE}",
                                  "oof_slope": cal["cal_slope"]},
                "final_lasso_C": finfo.get("lasso_C"), "final_gbm_params": finfo["gbm_params"],
                "final_n_features": finfo["n_features"], "final_features": finfo["features"],
                "oof_auc": roc_auc_score(y, p_oof),
                "pipeline_file": path.name, "pipeline_sha256": sha256(path),
            }
            print(f"  OOF AUC {frozen[key]['oof_auc']:.3f}  threshold {thr:.3f}  "
                  f"slope {cal['cal_slope']:.2f}  final features {finfo['n_features']}  "
                  f"final fit {time.time() - t0:.0f}s", flush=True)

            if arm == "lasso":
                sel_tables.append(lasso_tables(v, x, y, folds, sel_rows, final, finfo,
                                               args, out))

    pd.concat(sel_tables).to_csv(out / "features" / "lasso_selection_frequency.csv",
                                 index=False)
    pd.DataFrame(fold_rows).to_csv(out / "cv" / "cv_fold_results.csv", index=False)
    summ = (pd.DataFrame(fold_rows).groupby(["variant", "arm"])
            [["validation_auc", "validation_log_loss", "validation_brier", "n_features"]]
            .agg(["mean", "std"]).round(4))
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.to_csv(out / "cv" / "cv_summary.csv")
    pd.concat(oof_rows).to_csv(out / "cv" / "oof_predictions.csv", index=False)

    # reproducibility: refit one outer fold from scratch, predictions must be identical
    if not args.skip_repro_check:
        v, arm = args.variants[0], "lasso"
        spec = gp.VARIANTS[v]
        x = train[spec["data"]][spec["features"]]
        tr, va = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
        pipe, _ = gp.fit_tuned(x.iloc[tr], y[tr], arm, inner, n_jobs=args.n_jobs,
                               n_iter=args.n_iter, Cs=args.lasso_cs)
        diff = np.max(np.abs(pipe.predict_proba(x.iloc[va])[:, 1] - oof[f"{v}_{arm}"][va]))
        qa.check(diff < 1e-12, f"Reproducible: refitting {v}_{arm} fold 0 gives identical "
                 f"predictions (max difference {diff:.1e})")

    frozen_doc = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Frozen before any test-set feature or outcome was read.",
        "seed": SEED, "outer_folds": OUTER_FOLDS, "inner_folds": gp.INNER_FOLDS,
        "folds_sha256": hashlib.sha256(folds.tobytes()).hexdigest(),
        "n_iter": args.n_iter, "lasso_Cs": list(map(float, args.lasso_cs)),
        "gbm_space": {k: [str(x) for x in v] for k, v in gp.GBM_SPACE.items()},
        "scoring": gp.SCORING, "n_boot": args.n_boot, "models": frozen,
        "versions": versions(),
    }
    (out / "models" / "frozen_decisions.json").write_text(json.dumps(frozen_doc, indent=2,
                                                                      default=str))
    oof_tables(out, oof, y, ids, args)
    qa.note("Stage 1 never read test-set features or outcomes (IDs only, for the overlap check)")


def lasso_tables(v, x, y, folds, sel_rows, final, finfo, args, out):
    """Selection frequency: outer folds + patient-level bootstrap at the final C."""
    rng = np.random.default_rng(SEED)
    C = finfo["lasso_C"]
    counts, n_ok = {}, 0
    for _ in range(args.lasso_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = gp.build_lasso_logistic(x, C).fit(x.iloc[idx], y[idx])
        names = [gp.clean_name(n) for n in m.named_steps["preprocess"].get_feature_names_out()]
        for n, c in zip(names, m.named_steps["l1"].coef_[0]):
            counts[n] = counts.get(n, 0) + int(c != 0)
        n_ok += 1
    sel = pd.DataFrame([r for r in sel_rows if r["variant"] == v])
    fold_freq = sel.groupby("feature")["fold"].nunique() / OUTER_FOLDS
    names = [gp.clean_name(n) for n in final.named_steps["preprocess"].get_feature_names_out()]
    coef = dict(zip(names, final.named_steps["lasso"].estimator_.coef_[0]))
    rows = []
    for n in names:
        base = n.replace("missingindicator_", "")
        rows.append({"variant": v, "feature": n,
                     "outer_fold_selection_freq": float(fold_freq.get(n, 0.0)),
                     "bootstrap_selection_freq": counts.get(n, 0) / n_ok,
                     "selected_in_final_model": coef[n] != 0,
                     "final_lasso_coef_per_SD": coef[n],
                     "final_lasso_C": C,
                     "timing_and_leakage": gp.TIMING_NOTES.get(base, "")})
    return pd.DataFrame(rows).sort_values(["bootstrap_selection_freq", "feature"],
                                          ascending=[False, True])


def perf_rows(label, v, arm, y, p, thr, ids, n_boot, recal=False):
    boot = ge.bootstrap_metrics(y, p, thr, ids, n_boot=n_boot, seed=SEED)
    cal = ge.calibration_stats(y, p)
    row = {"variant": v, "model": gp.VARIANTS[v]["label"], "arm": gp.ARMS[arm],
           "dataset": label, "recalibrated": recal, "n": len(y), "deaths": int(np.sum(y)),
           "threshold": thr, **boot.attrs["confusion"]}
    for _, r in boot.iterrows():
        row[r["metric"]] = r["estimate"]
        row[r["metric"] + "_low"] = r["ci_low"]
        row[r["metric"] + "_high"] = r["ci_high"]
    row.update({k: cal[k] for k in ("cal_intercept", "cal_intercept_low",
                                    "cal_intercept_high", "cal_slope", "cal_slope_low",
                                    "cal_slope_high", "log_loss")})
    row["bootstrap_resamples"] = boot.attrs["n_boot_used"]
    return row


def formatted(df):
    out = pd.DataFrame({
        "Model": df["model"], "Features": df["arm"], "Dataset": df["dataset"],
        "Recalibrated": df["recalibrated"].map({True: "yes", False: "no"}),
        "n (deaths)": df["n"].astype(str) + " (" + df["deaths"].astype(str) + ")",
        "Threshold": df["threshold"].map(lambda t: f"{t:.3f}")})
    for m, name in [("auc", "AUC (95% CI)"), ("sensitivity", "Sensitivity"),
                    ("specificity", "Specificity"), ("accuracy", "Accuracy"),
                    ("f1", "F1"), ("ppv", "PPV"), ("npv", "NPV"), ("brier", "Brier")]:
        out[name] = [fmt_ci(e, lo, hi) for e, lo, hi in
                     zip(df[m], df[m + "_low"], df[m + "_high"])]
    out["Calibration intercept"] = [fmt_ci(e, lo, hi, 2) for e, lo, hi in zip(
        df["cal_intercept"], df["cal_intercept_low"], df["cal_intercept_high"])]
    out["Calibration slope"] = [fmt_ci(e, lo, hi, 2) for e, lo, hi in zip(
        df["cal_slope"], df["cal_slope_low"], df["cal_slope_high"])]
    return out


def comparisons(preds, y, ids, n_boot, label):
    rows = []
    for v1, a1, v2, a2 in COMPARISONS:
        k1, k2 = f"{v1}_{a1}", f"{v2}_{a2}"
        if k1 in preds and k2 in preds:
            d = ge.paired_auc_difference(y, preds[k1], preds[k2], ids, n_boot, SEED)
            rows.append({"dataset": label,
                         "comparison": f"{gp.VARIANTS[v1]['label']} / {gp.ARMS[a1]}  minus  "
                                       f"{gp.VARIANTS[v2]['label']} / {gp.ARMS[a2]}",
                         "first": k1, "second": k2, **d})
    return pd.DataFrame(rows)


def oof_tables(out, oof, y, ids, args):
    frozen = json.loads((out / "models" / "frozen_decisions.json").read_text())["models"]
    rows = []
    for key, p in oof.items():
        fz = frozen[key]
        rows.append(perf_rows("Training out-of-fold", fz["variant"], fz["arm"], y, p,
                              fz["threshold_youden_oof"], ids, args.n_boot))
    df = pd.DataFrame(rows)
    df.to_csv(out / "tables" / "performance_train_oof_full.csv", index=False)
    formatted(df).to_csv(out / "tables" / "performance_train_oof.csv", index=False)
    comparisons(oof, y, ids, args.n_boot, "Training out-of-fold").to_csv(
        out / "tables" / "auc_comparisons_train_oof.csv", index=False)
    make_figures(out, {"Training out-of-fold": (y, oof)}, [k for k in oof])


def make_figures(out, datasets, keys):
    """datasets: {panel label: (y, {key: p})}"""
    fig_dir = out / "figures"
    variants = sorted({k.rsplit("_", 1)[0] for k in keys}, key=list(gp.VARIANTS).index)
    for v in variants:
        title = gp.VARIANTS[v]["label"]
        arms = [a for a in gp.ARMS if f"{v}_{a}" in keys]
        ge.plot_roc([(lab, y, {a: p[f"{v}_{a}"] for a in arms})
                     for lab, (y, p) in datasets.items()], title, fig_dir / f"roc_{v}")
        ge.plot_dca([(lab, y, {a: p[f"{v}_{a}"] for a in arms})
                     for lab, (y, p) in datasets.items()], title, fig_dir / f"dca_{v}")
        for a in arms:
            ge.plot_calibration([(lab, y, p[f"{v}_{a}"], ge.ARM_COLORS[a])
                                 for lab, (y, p) in datasets.items()],
                                f"{title} - {gp.ARMS[a]}", fig_dir / f"calibration_{v}_{a}")


# --------------------------------------------------------------------------- #
# Stage 2: the single test evaluation
# --------------------------------------------------------------------------- #
def stage2(args, out, qa):
    lock = out / "test" / "TEST_EVALUATED.lock"
    if lock.exists() and not args.overwrite_test:
        raise SystemExit(f"Test set already evaluated ({lock.read_text().strip()}). "
                         "Re-running would be a second look; pass --overwrite-test only "
                         "to reproduce the same frozen models.")
    doc = json.loads((out / "models" / "frozen_decisions.json").read_text())
    frozen = doc["models"]
    oof_df = pd.read_csv(out / "cv" / "oof_predictions.csv")
    (out / "test").mkdir(exist_ok=True)

    test = {d: read_frame(args.data_dir / f"{d}_test.csv") for d in ("A", "B")}
    qa.check(test["A"][ID].equals(test["B"][ID]) and test["A"][TARGET].equals(test["B"][TARGET]),
             "Model A and B test files contain the same patients and outcomes")
    train_ids = set(oof_df[ID])
    qa.check(not train_ids & set(test["A"][ID]), "No test patient was used in training")
    yt = test["A"][TARGET].to_numpy().astype(int)
    tid = test["A"][ID].to_numpy()

    pred_rows, perf, test_p, oof_p = [], [], {}, {}
    y_oof = None
    for key, fz in frozen.items():
        v, arm = fz["variant"], fz["arm"]
        path = out / "models" / fz["pipeline_file"]
        qa.check(sha256(path) == fz["pipeline_sha256"],
                 f"{key}: pipeline file unchanged since it was frozen")
        pipe = joblib.load(path)
        xt = test[gp.VARIANTS[v]["data"]][fz["features_candidates"]]
        p = pipe.predict_proba(xt)[:, 1]
        thr = fz["threshold_youden_oof"]
        test_p[key] = p
        o = oof_df[(oof_df["variant"] == v) & (oof_df["arm"] == arm)].sort_values(ID)
        oof_p[key] = o["predicted"].to_numpy()
        y_oof, ids_oof = o["observed"].to_numpy(), o[ID].to_numpy()
        perf.append(perf_rows("Training out-of-fold", v, arm, y_oof, oof_p[key], thr,
                              ids_oof, args.n_boot))
        perf.append(perf_rows("Held-out test", v, arm, yt, p, thr, tid, args.n_boot))
        rec = pd.DataFrame({ID: tid, "variant": v, "arm": arm, "observed": yt,
                            "predicted_probability": p, "threshold": thr,
                            "predicted_class": (p >= thr).astype(int)})
        if fz["recalibration"]["applied"]:
            r = fz["recalibration"]
            p_r = ge.apply_recalibration(p, r["a"], r["b"])
            # threshold re-derived on recalibrated OOF predictions (training data only)
            oof_r = ge.apply_recalibration(oof_p[key], r["a"], r["b"])
            thr_r = ge.youden_threshold(y_oof, oof_r)
            perf.append(perf_rows("Training out-of-fold", v, arm, y_oof, oof_r, thr_r,
                                  ids_oof, args.n_boot, recal=True))
            perf.append(perf_rows("Held-out test", v, arm, yt, p_r, thr_r, tid, args.n_boot,
                                  recal=True))
            rec["recalibrated_probability"] = p_r
        pred_rows.append(rec)
        print(f"  TEST {key}: AUC {roc_auc_score(yt, p):.3f}", flush=True)

    pd.concat(pred_rows).to_csv(out / "test" / "test_predictions.csv", index=False)
    df = pd.DataFrame(perf)
    df.to_csv(out / "tables" / "performance_full.csv", index=False)
    formatted(df).to_csv(out / "tables" / "performance_train_oof_and_test.csv", index=False)
    pd.concat([comparisons(oof_p, y_oof, ids_oof, args.n_boot, "Training out-of-fold"),
               comparisons(test_p, yt, tid, args.n_boot, "Held-out test")]).to_csv(
        out / "tables" / "auc_comparisons.csv", index=False)
    make_figures(out, {"Training out-of-fold": (y_oof, oof_p), "Held-out test": (yt, test_p)},
                 list(frozen))
    lock.write_text(f"evaluated {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    qa.note("Test outcomes were used only after every threshold, recalibration and "
            "hyperparameter was frozen (frozen_decisions.json written in stage 1)")


def versions():
    import matplotlib
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__}


def main():
    project = SCRIPTS.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=project / "splits")
    ap.add_argument("--output-dir", type=Path, default=project / "results" / "gbm")
    ap.add_argument("--variants", default="A,B,B_noscores")
    ap.add_argument("--final-test", action="store_true",
                    help="Stage 2: evaluate the frozen models once on the test set")
    ap.add_argument("--overwrite-test", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--lasso-boot", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="tiny grids, for smoke tests only")
    ap.add_argument("--skip-repro-check", action="store_true")
    args = ap.parse_args()
    args.variants = [v.strip() for v in args.variants.split(",")]
    args.n_iter, args.lasso_cs = gp.GBM_N_ITER, gp.LASSO_CS
    if args.quick:
        args.n_iter, args.lasso_cs = 3, np.logspace(-2, 0, 5)
        args.n_boot, args.lasso_boot = 200, 10
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    qa = QA()
    t0 = time.time()
    if args.final_test:
        print("STAGE 2 - single evaluation on the held-out test set", flush=True)
        stage2(args, out, qa)
        qa.write(out / "qa_report_stage2_test.txt")
    else:
        print("STAGE 1 - training set only (test features and outcomes not read)", flush=True)
        stage1(args, out, qa)
        qa.write(out / "qa_report_stage1_training.txt")
        print("\nTest set untouched. Run with --final-test once these results are accepted.")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {out}")


if __name__ == "__main__":
    main()
