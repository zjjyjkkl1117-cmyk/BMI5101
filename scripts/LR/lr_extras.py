"""P15: supplementary analyses for the logistic regression model (lr_cv.py).

Calibration (raw / temperature scaling / isotonic regression), decision curve
analysis, SHAP, and the temporal-feature and leakage comparisons (Model B with vs
without early_/late_/slope_/sd_ features; Model A vs Model B). Same split, folds,
pipeline and frozen LASSO C as lr_cv.py. Recalibration maps are learned on
training out-of-fold probabilities (shared 5-fold splits) only.

Usage (from project root, after `python scripts/LR/lr_cv.py --stage tune`):
    python scripts/LR/lr_extras.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path[:0] = [str(Path(__file__).resolve().parent),  # scripts/LR (lr_cv.py)
                str(Path(__file__).resolve().parents[1])]  # scripts (metrics.py)
from lr_cv import (DROP_FEATURES, FOLDS, ID, SEED, TARGET,  # noqa: E402
                   build_pipeline, get_folds, load_best_config, load_data, output_dir)

TEMPORAL_PREFIXES = ("early_", "late_", "slope_", "sd_")
N_BOOT = 2000
EPS = 1e-6
DCA_THRESHOLDS = np.round(np.arange(0.01, 0.61, 0.01), 2)

# Chart styling (reference palette; categorical slots used in fixed order)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def new_figure(ncols=1, width=6.0, height=4.6):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        style_axes(ax)
    return fig, axes.ravel()


def fit_cv_and_final(x, y, x_test, folds, lasso_c):
    """Out-of-fold training probabilities, then a final fit on all training data."""
    oof = np.full(len(y), np.nan)
    fold_auc = []
    for fold in range(FOLDS):
        ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        pipe = build_pipeline(x, lasso_c=lasso_c).fit(x.iloc[ti], y[ti])
        oof[vi] = pipe.predict_proba(x.iloc[vi])[:, 1]
        fold_auc.append(roc_auc_score(y[vi], oof[vi]))
    final = build_pipeline(x, lasso_c=lasso_c).fit(x, y)
    return oof, final.predict_proba(x_test)[:, 1], final, fold_auc


# ---------- calibration ----------

def fit_temperature(p, y):
    z = logit(np.clip(p, EPS, 1 - EPS))
    res = minimize_scalar(lambda t: log_loss(y, expit(z / t)), bounds=(0.05, 20), method="bounded")
    return float(res.x)


def calibration_intercept_slope(p, y):
    z = logit(np.clip(p, EPS, 1 - EPS))
    slope = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit().params[1]
    intercept = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                       offset=z).fit().params[0]
    return float(intercept), float(slope)


def calibration_bins(p, y, n_bins=10):
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"),
                                 n=("y", "size")).reset_index()


# ---------- decision curve ----------

def net_benefit(y, p, pt):
    pred = p >= pt
    n = len(y)
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return tp / n - fp / n * pt / (1 - pt)


# ---------- comparison ----------

def paired_bootstrap_auc(y, p1, p2, seed=SEED):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two_sided = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return float(roc_auc_score(y, p1) - roc_auc_score(y, p2)), float(lo), float(hi), float(p_two_sided)


def clean_name(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


def main():
    project = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=project / "splits")
    p.add_argument("--output-dir", type=Path, default=project / "results" / "lr" / "extras")
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    a_train, a_test = load_data(args.data_dir, "A")
    b_train, b_test = load_data(args.data_dir, "B")
    folds = get_folds(a_train, args.data_dir / "cv_folds.csv")
    y = a_train[TARGET].to_numpy()
    y_test = a_test[TARGET].to_numpy()

    a_cols = [c for c in a_train.columns if c not in DROP_FEATURES]
    b_cols = [c for c in b_train.columns if c not in DROP_FEATURES]
    lr_out = output_dir(project, clip=True)
    c_a = load_best_config(lr_out, "A")["lasso_C"]
    c_b = load_best_config(lr_out, "B")["lasso_C"]
    # name: (train, test, columns, frozen LASSO C from lr_cv.py --stage tune)
    feature_sets = {
        "LR_A": (a_train, a_test, a_cols, c_a),
        "LR_B": (b_train, b_test, b_cols, c_b),
        "LR_B_no_temporal": (b_train, b_test,
                             [c for c in b_cols if not c.startswith(TEMPORAL_PREFIXES)], c_b),
    }
    fits = {}
    for name, (tr, te, cols, lasso_c) in feature_sets.items():
        print(f"Fitting {name} ({len(cols)} raw predictors, LASSO C={lasso_c}) ...", flush=True)
        oof, p_te, final, fold_auc = fit_cv_and_final(tr[cols], y, te[cols], folds, lasso_c)
        fits[name] = {"oof": oof, "test": p_te, "final": final, "fold_auc": fold_auc,
                      "cols": cols, "x_train": tr[cols], "x_test": te[cols]}

    # 1) Calibration: raw vs temperature scaling vs isotonic, learned on training OOF
    cal_rows, bin_rows, temps = [], [], {}
    fig, axes = new_figure(ncols=2)
    for ax, name in zip(axes, ["LR_A", "LR_B"]):
        oof, p_te = fits[name]["oof"], fits[name]["test"]
        t = fit_temperature(oof, y)
        temps[name] = t
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(oof, y)
        versions = {
            "Raw": p_te,
            "Temperature scaling": expit(logit(np.clip(p_te, EPS, 1 - EPS)) / t),
            "Isotonic regression": iso.predict(p_te),
        }
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", label="Perfect calibration")
        for color, (method, prob) in zip(SERIES, versions.items()):
            intercept, slope = calibration_intercept_slope(prob, y_test)
            cal_rows.append({"model": name, "method": method, "temperature": t if method.startswith("Temp") else np.nan,
                             "test_auroc": roc_auc_score(y_test, prob),
                             "test_brier": brier_score_loss(y_test, prob),
                             "calibration_intercept": intercept, "calibration_slope": slope,
                             "mean_predicted": float(np.mean(prob)),
                             "observed_rate": float(np.mean(y_test))})
            bins = calibration_bins(prob, y_test)
            bin_rows.append(bins.assign(model=name, method=method))
            ax.plot(bins["mean_predicted"], bins["observed_rate"], color=color, linewidth=2,
                    marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, label=method)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability (test, deciles)")
        ax.set_ylabel("Observed mortality")
        ax.set_title(f"{name}: calibration on test set (T = {t:.2f})", fontsize=11)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "calibration_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    pd.DataFrame(cal_rows).to_csv(out / "calibration_metrics.csv", index=False)
    pd.concat(bin_rows).to_csv(out / "calibration_bins.csv", index=False)

    # 2) Decision curve analysis on the test set (raw probabilities)
    prevalence = y_test.mean()
    dca = pd.DataFrame({"threshold": DCA_THRESHOLDS})
    dca["treat_all"] = prevalence - (1 - prevalence) * dca["threshold"] / (1 - dca["threshold"])
    dca["treat_none"] = 0.0
    for name in ["LR_A", "LR_B"]:
        dca[name] = [net_benefit(y_test, fits[name]["test"], t) for t in DCA_THRESHOLDS]
    dca.to_csv(out / "decision_curve.csv", index=False)
    fig, (ax,) = new_figure()
    for color, name in zip(SERIES, ["LR_A", "LR_B"]):
        ax.plot(dca["threshold"], dca[name], color=color, linewidth=2, label=name)
    ax.plot(dca["threshold"], dca["treat_all"], color=MUTED, linewidth=1.5, linestyle="--", label="Treat all")
    ax.plot(dca["threshold"], dca["treat_none"], color=INK2, linewidth=1.5, linestyle=":", label="Treat none")
    ax.set_ylim(-0.05, max(0.25, float(dca[["LR_A", "LR_B"]].max().max()) + 0.02))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis (test set)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "decision_curve.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    dca_summary = {}
    for name in ["LR_A", "LR_B"]:
        better = dca["threshold"][dca[name] > np.maximum(dca["treat_all"], dca["treat_none"])]
        dca_summary[name] = [float(better.min()), float(better.max())] if len(better) else None

    # 3) SHAP for the final models (linear explainer; background = training patients)
    shap_summary = {}
    for name in ["LR_A", "LR_B"]:
        final = fits[name]["final"]
        to_model = final[:-1]  # preprocess + LASSO selection
        x_bg = to_model.transform(fits[name]["x_train"])
        x_te = to_model.transform(fits[name]["x_test"])
        names = [clean_name(n) for n in to_model.get_feature_names_out()]
        lr = final.named_steps["lr"]
        explainer = shap.LinearExplainer((lr.coef_.ravel(), float(lr.intercept_[0])), x_bg)
        values = explainer.shap_values(x_te)
        abs_vals = np.abs(values)
        # A few test patients have values far outside the training range (e.g. extreme
        # slope_* artefacts), which dominate the plain mean. The trimmed mean drops the
        # top 1% of |SHAP| per feature (about 4 patients) and is used for ranking.
        cut = np.quantile(abs_vals, 0.99, axis=0)
        trimmed = np.where(abs_vals <= cut, abs_vals, np.nan)
        imp = pd.DataFrame({"feature": names,
                            "trimmed_mean_abs_shap": np.nanmean(trimmed, axis=0),
                            "mean_abs_shap": abs_vals.mean(axis=0),
                            "max_abs_shap": abs_vals.max(axis=0),
                            "coefficient": lr.coef_.ravel()}) \
            .sort_values("trimmed_mean_abs_shap", ascending=False)
        imp.to_csv(out / f"shap_importance_{name}.csv", index=False)
        shap_summary[name] = imp["feature"].head(10).tolist()

        top = imp.head(15).iloc[::-1]
        fig, (ax,) = new_figure(width=6.5, height=5.2)
        ax.barh(top["feature"], top["trimmed_mean_abs_shap"], color=SERIES[0], height=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Trimmed mean |SHAP| (log-odds), test set")
        ax.set_title(f"{name}: top 15 features by SHAP", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"shap_bar_{name}.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)

        order = imp.index.to_numpy()[:15]  # same ranking as the bar chart
        lim = float(np.quantile(abs_vals[:, order], 0.995)) * 1.1
        n_hidden = int((abs_vals[:, order] > lim).sum())
        plt.figure()
        shap.summary_plot(values[:, order], x_te[:, order], feature_names=[names[i] for i in order],
                          max_display=15, sort=False, show=False)
        plt.xlim(-lim, lim)
        plt.title(f"{name}: SHAP values (test set; {n_hidden} extreme points beyond axis)", fontsize=10)
        plt.tight_layout()
        plt.savefig(out / f"shap_beeswarm_{name}.png", dpi=160)
        plt.close("all")

    # 4) Comparisons: temporal features and leakage
    comp_rows = []
    for label, m1, m2 in [("Temporal features: LR_B vs LR_B_no_temporal", "LR_B", "LR_B_no_temporal"),
                          ("Leaky variables: LR_A vs LR_B", "LR_A", "LR_B")]:
        for cohort, yy, key in [("Train (5-fold CV, pooled OOF)", y, "oof"), ("Test", y_test, "test")]:
            diff, lo, hi, pval = paired_bootstrap_auc(yy, fits[m1][key], fits[m2][key])
            comp_rows.append({"comparison": label, "cohort": cohort,
                              "auroc_1": roc_auc_score(yy, fits[m1][key]),
                              "auroc_2": roc_auc_score(yy, fits[m2][key]),
                              "difference": diff, "ci_low": lo, "ci_high": hi,
                              "bootstrap_p": pval})
    pd.DataFrame(comp_rows).to_csv(out / "model_comparisons.csv", index=False)

    pd.DataFrame({name: fits[name]["fold_auc"] for name in fits}).rename_axis("fold") \
        .reset_index().to_csv(out / "cv_fold_auroc_by_model.csv", index=False)

    (out / "run_summary.json").write_text(json.dumps({
        "temperatures": temps, "dca_threshold_range_model_beats_defaults": dca_summary,
        "shap_top10": shap_summary,
        "n_raw_predictors": {k: len(v["cols"]) for k, v in fits.items()},
        "cv_mean_fold_auroc": {k: float(np.mean(v["fold_auc"])) for k, v in fits.items()},
    }, indent=2), encoding="utf-8")

    pd.set_option("display.width", 160)
    print("\nCalibration (test set):")
    print(pd.DataFrame(cal_rows).round(3).to_string(index=False))
    print("\nModel comparisons (AUROC difference, paired bootstrap 95% CI):")
    print(pd.DataFrame(comp_rows).round(3).to_string(index=False))
    print("\nDCA: threshold range where model beats treat-all and treat-none:", dca_summary)
    print("SHAP top 10:", json.dumps(shap_summary, indent=1))
    print("Saved to", out.name)


if __name__ == "__main__":
    main()
