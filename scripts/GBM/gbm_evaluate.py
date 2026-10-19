"""P15 GBM - evaluation: threshold, metrics, bootstrap CIs, calibration, decision curves, figures.

Methods (also in README):
  * Threshold: Youden's J (max sensitivity + specificity - 1) on training out-of-fold
    predictions; a patient is predicted positive when p >= threshold.
  * 95% CIs: patient-level bootstrap, 2,000 resamples, percentile intervals. The
    threshold is held fixed inside the bootstrap (it is not re-optimised).
  * Calibration intercept: logistic regression of the outcome on an intercept with
    logit(p) as offset (calibration-in-the-large; ideal 0).
    Calibration slope: logistic regression of the outcome on logit(p) (ideal 1).
  * Calibration curve: deciles of predicted risk (observed rate with Wilson 95% CI)
    plus a flexible curve (logistic regression on a cubic B-spline of logit(p)).
  * Net benefit = TP/n - FP/n * pt / (1 - pt); treat-all = prevalence -
    (1 - prevalence) * pt / (1 - pt); treat-none = 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

EPS = 1e-6
DCA_THRESHOLDS = np.round(np.arange(0.05, 0.501, 0.01), 2)
POINT_METRICS = ["auc", "sensitivity", "specificity", "accuracy", "f1", "ppv", "npv", "brier"]


# --------------------------------------------------------------------------- #
# Threshold and point metrics
# --------------------------------------------------------------------------- #
def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j[1:]) + 1)       # index 0 is the +inf "predict nobody" point
    return float(thr[i])


def point_metrics(y, p, threshold) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else np.nan
    return {
        "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "accuracy": div(tp + tn, len(y)),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "brier": float(np.mean((p - y) ** 2)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _resample_index(groups, rng, n_boot):
    """Yield row indices for patient-level bootstrap resamples."""
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    if len(uniq) == len(groups):                # one row per patient
        for _ in range(n_boot):
            yield rng.integers(0, len(groups), len(groups))
    else:
        members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            yield np.concatenate([members[g] for g in pick])


def bootstrap_metrics(y, p, threshold, groups, n_boot=2000, seed=42) -> pd.DataFrame:
    """Point estimate + percentile 95% CI for every metric (threshold fixed)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for idx in _resample_index(groups, rng, n_boot):
        if len(np.unique(y[idx])) < 2:
            continue
        m = point_metrics(y[idx], p[idx], threshold)
        draws.append([m[k] for k in POINT_METRICS])
    draws = np.array(draws)
    est = point_metrics(y, p, threshold)
    rows = []
    for j, k in enumerate(POINT_METRICS):
        lo, hi = np.nanpercentile(draws[:, j], [2.5, 97.5])
        rows.append({"metric": k, "estimate": est[k], "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    out.attrs["n_boot_used"] = len(draws)
    out.attrs["confusion"] = {k: est[k] for k in ("tp", "fp", "tn", "fn")}
    return out


def paired_auc_difference(y, p1, p2, groups, n_boot=2000, seed=42) -> dict:
    """AUC(p1) - AUC(p2) on the same patients, paired patient-level bootstrap."""
    y = np.asarray(y).astype(int)
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    rng = np.random.default_rng(seed)
    diffs = []
    for idx in _resample_index(groups, rng, n_boot):
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    diffs = np.array(diffs)
    est = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value: share of resamples on the other side of zero
    p_val = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"delta_auc": est, "ci_low": lo, "ci_high": hi, "bootstrap_p": p_val}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _logistic_fit(X, y, offset):
    """Unpenalised logistic regression by Newton-Raphson (supports an offset)."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        mu = expit(X @ beta + offset)
        w = np.clip(mu * (1 - mu), 1e-12, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = expit(X @ beta + offset)
    cov = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    return beta, np.sqrt(np.diag(cov))


def calibration_stats(y, p) -> dict:
    y = np.asarray(y, float)
    lp = logit(np.clip(np.asarray(p, float), EPS, 1 - EPS))
    ones = np.ones((len(y), 1))
    (a,), (a_se,) = _logistic_fit(ones, y, lp)
    (b0, b1), (_, b1_se) = _logistic_fit(np.column_stack([ones, lp]), y, np.zeros(len(y)))
    z = norm.ppf(0.975)
    return {
        "brier": float(np.mean((np.asarray(p) - y) ** 2)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS))),
        "cal_intercept": float(a), "cal_intercept_low": float(a - z * a_se),
        "cal_intercept_high": float(a + z * a_se),
        "cal_slope": float(b1), "cal_slope_low": float(b1 - z * b1_se),
        "cal_slope_high": float(b1 + z * b1_se),
        "recal_a": float(b0), "recal_b": float(b1),
    }


def apply_recalibration(p, a, b):
    lp = logit(np.clip(np.asarray(p, float), EPS, 1 - EPS))
    return expit(a + b * lp)


def _wilson(k, n, z=1.96):
    phat = k / n
    den = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def binned_calibration(y, p, n_bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed=("y", "mean"),
                              events=("y", "sum"), n=("y", "size")).reset_index()
    g["ci_low"], g["ci_high"] = _wilson(g["events"].to_numpy(), g["n"].to_numpy())
    return g


def flexible_calibration_curve(y, p, n_grid=200):
    lp = logit(np.clip(np.asarray(p, float), EPS, 1 - EPS)).reshape(-1, 1)
    model = make_pipeline(SplineTransformer(n_knots=4, degree=3, extrapolation="linear"),
                          LogisticRegression(C=1e6, max_iter=5000))
    model.fit(lp, np.asarray(y, int))
    lo, hi = np.quantile(np.asarray(p, float), [0.01, 0.99])
    grid = np.linspace(lo, hi, n_grid)
    obs = model.predict_proba(logit(np.clip(grid, EPS, 1 - EPS)).reshape(-1, 1))[:, 1]
    return grid, obs


# --------------------------------------------------------------------------- #
# Decision curve analysis
# --------------------------------------------------------------------------- #
def decision_curve(y, p, thresholds=DCA_THRESHOLDS) -> pd.DataFrame:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    n, prev = len(y), y.mean()
    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        odds = pt / (1 - pt)
        rows.append({"threshold": pt, "net_benefit_model": tp / n - fp / n * odds,
                     "net_benefit_treat_all": prev - (1 - prev) * odds,
                     "net_benefit_treat_none": 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures (matplotlib, light publication style)
# --------------------------------------------------------------------------- #
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e0"
ARM_COLORS = {"lasso": "#2a78d6", "all": "#eb6834"}    # categorical slots 1, 2


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, path_stem):
    fig.savefig(f"{path_stem}.png")
    fig.savefig(f"{path_stem}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_roc(panels, title, path_stem):
    """panels: list of (panel_title, y, {arm: p})."""
    import matplotlib.pyplot as plt
    set_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.3),
                             squeeze=False)
    for ax, (ptitle, y, probs) in zip(axes[0], panels):
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Chance")
        for arm, p in probs.items():
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=ARM_COLORS[arm],
                    label=f"{ARM_LABEL[arm]} (AUC {roc_auc_score(y, p):.3f})")
        ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="1 - specificity",
               ylabel="Sensitivity", title=ptitle)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    save(fig, path_stem)


ARM_LABEL = {"lasso": "GBM + LASSO", "all": "GBM, all features"}


def plot_calibration(panels, title, path_stem):
    """panels: list of (panel_title, y, p, color)."""
    import matplotlib.pyplot as plt
    set_style()
    fig = plt.figure(figsize=(4.6 * len(panels), 5.2))
    gs = fig.add_gridspec(2, len(panels), height_ratios=[4.2, 1], hspace=0.08, wspace=0.3)
    for j, (ptitle, y, p, color) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        hx = fig.add_subplot(gs[1, j], sharex=ax)
        stats = calibration_stats(y, p)
        bins = binned_calibration(y, p)
        grid, obs = flexible_calibration_curve(y, p)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Ideal calibration")
        ax.plot(grid, obs, color=color, label="Flexible calibration curve")
        ax.errorbar(bins["mean_predicted"], bins["observed"],
                    yerr=[bins["observed"] - bins["ci_low"], bins["ci_high"] - bins["observed"]],
                    fmt="o", ms=5, color=color, mfc="white", mew=1.6, elinewidth=1,
                    capsize=0, label="Deciles of predicted risk (95% CI)")
        ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed proportion of deaths",
               title=ptitle)
        ax.tick_params(labelbottom=False)
        txt = (f"Brier {stats['brier']:.3f}\n"
               f"Intercept {stats['cal_intercept']:.2f} "
               f"({stats['cal_intercept_low']:.2f} to {stats['cal_intercept_high']:.2f})\n"
               f"Slope {stats['cal_slope']:.2f} "
               f"({stats['cal_slope_low']:.2f} to {stats['cal_slope_high']:.2f})")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, color=INK)
        ax.legend(loc="lower right", fontsize=8)
        y = np.asarray(y, int)
        edges = np.linspace(0, 1, 41)
        hx.hist([np.asarray(p)[y == 0], np.asarray(p)[y == 1]], bins=edges,
                stacked=True, color=["#c9c8c3", color], edgecolor="white",
                linewidth=0.5, label=["Survived", "Died"])
        hx.set(xlabel="Predicted probability of in-hospital death", ylabel="Patients")
        hx.grid(False)
        hx.legend(loc="upper right", fontsize=7.5)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    save(fig, path_stem)


def plot_dca(panels, title, path_stem):
    """panels: list of (panel_title, y, {arm: p})."""
    import matplotlib.pyplot as plt
    set_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.0),
                             squeeze=False)
    for ax, (ptitle, y, probs) in zip(axes[0], panels):
        prev = float(np.mean(y))
        base = decision_curve(y, next(iter(probs.values())))
        ax.plot(base["threshold"], base["net_benefit_treat_all"], color=MUTED,
                lw=1.5, label="Treat all")
        ax.axhline(0, color=INK, lw=1, ls=":", label="Treat none")
        for arm, p in probs.items():
            d = decision_curve(y, p)
            ax.plot(d["threshold"], d["net_benefit_model"], color=ARM_COLORS[arm],
                    label=ARM_LABEL[arm])
        ax.set(xlim=(DCA_THRESHOLDS[0], DCA_THRESHOLDS[-1]),
               ylim=(-0.05, max(0.05, prev + 0.03)),
               xlabel="Threshold probability", ylabel="Net benefit", title=ptitle)
        ax.legend(loc="upper right", fontsize=8.5)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    save(fig, path_stem)
