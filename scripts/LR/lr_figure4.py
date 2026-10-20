"""P15: calibration and decision curves for LR in the format of the paper's Figure 4.

(A) training-cohort calibration, (B) training-cohort DCA, (C) test-cohort calibration,
(D) test-cohort DCA. Training cohort = apparent predictions of the final model (as in
the paper); test cohort = held-out predictions. Calibration uses three equal-width
probability bins (0-33%, 33-67%, 67-100%) plotted at their midpoints, as in the paper;
DCA shows standardized net benefit (net benefit / prevalence).

Reads results/lr/train_predictions_*.csv and test_predictions_*.csv written by
`python scripts/LR/lr_cv.py --stage final`; nothing is refitted.

Usage (from project root):
    python scripts/LR/lr_figure4.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {"Model A": "#2a78d6", "Model B": "#eb6834"}
BIN_EDGES = np.array([0, 1 / 3, 2 / 3, 1.0])
THRESHOLDS = np.round(np.arange(0.0, 1.0, 0.01), 2)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)


def calibration_bins(y, p):
    idx = np.clip(np.digitize(p, BIN_EDGES[1:-1]), 0, 2)
    rows = []
    for b in range(3):
        mask = idx == b
        rows.append({"bin": b, "bin_midpoint_pct": 100 * (BIN_EDGES[b] + BIN_EDGES[b + 1]) / 2,
                     "n": int(mask.sum()),
                     "mean_predicted_pct": 100 * p[mask].mean() if mask.any() else np.nan,
                     "observed_pct": 100 * y[mask].mean() if mask.any() else np.nan})
    return pd.DataFrame(rows)


def standardized_net_benefit(y, p, t):
    prev = y.mean()
    pred = p >= t
    n = len(y)
    nb = np.sum(pred & (y == 1)) / n - np.sum(pred & (y == 0)) / n * t / (1 - t)
    return nb / prev


def main():
    project = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-dir", type=Path, default=project / "results" / "lr")
    args = ap.parse_args()
    out = args.lr_dir / "extras"
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for label, name in [("Model A", "A"), ("Model B", "B")]:
        tr = pd.read_csv(args.lr_dir / f"train_predictions_{name}.csv")
        te = pd.read_csv(args.lr_dir / f"test_predictions_{name}.csv")
        data[label] = {"Training cohort": (tr["observed"].to_numpy(), tr["apparent_probability"].to_numpy()),
                       "Test cohort": (te["observed"].to_numpy(), te["predicted_probability"].to_numpy())}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    fig.patch.set_facecolor(SURFACE)
    bin_rows, dca_rows = [], []
    for row, cohort in enumerate(["Training cohort", "Test cohort"]):
        ax_cal, ax_dca = axes[row]
        for ax in (ax_cal, ax_dca):
            style(ax)
        ax_cal.plot([0, 100], [0, 100], color=MUTED, linewidth=1.2, linestyle="--",
                    label="Perfect calibration")
        y_ref = None
        for label, cohorts in data.items():
            y, p = cohorts[cohort]
            y_ref = y
            bins = calibration_bins(y, p)
            bin_rows.append(bins.assign(model=f"LR {label}", cohort=cohort))
            ok = bins["n"] > 0
            ax_cal.plot(bins.loc[ok, "bin_midpoint_pct"], bins.loc[ok, "observed_pct"],
                        color=COLORS[label], linewidth=2, marker="o", markersize=6,
                        markeredgecolor=SURFACE, label=f"LR {label}")
            snb = [standardized_net_benefit(y, p, t) for t in THRESHOLDS]
            dca_rows.append(pd.DataFrame({"model": f"LR {label}", "cohort": cohort,
                                          "threshold": THRESHOLDS, "standardized_net_benefit": snb}))
            ax_dca.plot(THRESHOLDS, snb, color=COLORS[label], linewidth=2, label=f"LR {label}")
        prev = y_ref.mean()
        treat_all = (prev - (1 - prev) * THRESHOLDS / (1 - THRESHOLDS)) / prev
        ax_dca.plot(THRESHOLDS, treat_all, color=MUTED, linewidth=1.5, linestyle="--", label="All")
        ax_dca.axhline(0, color=INK2, linewidth=1.2, linestyle=":", label="None")
        dca_rows.append(pd.DataFrame({"model": "All", "cohort": cohort, "threshold": THRESHOLDS,
                                      "standardized_net_benefit": treat_all}))

        panel_cal, panel_dca = ("A", "B") if row == 0 else ("C", "D")
        ax_cal.set_xlim(0, 100)
        ax_cal.set_ylim(0, 100)
        ax_cal.set_xlabel("Bin midpoint (predicted risk, %)", color=INK2)
        ax_cal.set_ylabel("Observed event percentage", color=INK2)
        ax_cal.set_title(f"({panel_cal}) {cohort}: calibration", color=INK, fontsize=12)
        ax_cal.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")
        ax_dca.set_xlim(0, 1)
        ax_dca.set_ylim(-0.1, 1.0)
        ax_dca.set_xlabel("High risk threshold", color=INK2)
        ax_dca.set_ylabel("Standardized net benefit", color=INK2)
        ax_dca.set_title(f"({panel_dca}) {cohort}: decision curve", color=INK, fontsize=12)
        ax_dca.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out / "figure4_calibration_dca.png", dpi=170, facecolor=SURFACE)
    bins = pd.concat(bin_rows)
    bins.to_csv(out / "figure4_calibration_bins.csv", index=False)
    pd.concat(dca_rows).to_csv(out / "figure4_dca.csv", index=False)
    print(bins[["model", "cohort", "bin_midpoint_pct", "n", "mean_predicted_pct", "observed_pct"]]
          .round(1).to_string(index=False))
    print("Saved figure4_calibration_dca.png")


if __name__ == "__main__":
    main()
