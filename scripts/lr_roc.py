"""P15: ROC curves for the logistic regression models (training and test cohorts).

Refits the final lr_cv.py pipeline (with clipping) on all training patients and
plots ROC curves for Model A and Model B, in the format of the paper's Figure 3:
(A) training cohort (apparent), (B) test cohort. AUROC 95% CIs use metrics.py.

Usage (from project root):
    python scripts/lr_roc.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lr_cv import DROP_FEATURES, TARGET, build_pipeline, load_data  # noqa: E402
from metrics import auroc_ci  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {"Model A": "#2a78d6", "Model B": "#eb6834"}


def main():
    project = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=project / "splits")
    p.add_argument("--output-dir", type=Path, default=project / "results" / "lr" / "extras")
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    curves, rows = {"Training cohort": {}, "Test cohort": {}}, []
    for label, name in [("Model A", "A"), ("Model B", "B")]:
        tr, te = load_data(args.data_dir, name)
        cols = [c for c in tr.columns if c not in DROP_FEATURES]
        final = build_pipeline(tr[cols]).fit(tr[cols], tr[TARGET])
        for cohort, df in [("Training cohort", tr), ("Test cohort", te)]:
            y = df[TARGET].to_numpy()
            prob = final.predict_proba(df[cols])[:, 1]
            fpr, tpr, _ = roc_curve(y, prob)
            auc, lo, hi = auroc_ci(y, prob)
            curves[cohort][label] = (fpr, tpr, auc, lo, hi)
            rows.append({"model": f"LR {label}", "cohort": cohort, "auroc": auc,
                         "ci_low": lo, "ci_high": hi})
            pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
                out / f"roc_points_{name}_{cohort.split()[0].lower()}.csv", index=False)
        print(f"{label}: fitted", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, (panel, (cohort, models)) in zip(axes, zip("AB", curves.items())):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK2)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--")
        for label, (fpr, tpr, auc, lo, hi) in models.items():
            ax.plot(fpr, tpr, color=COLORS[label], linewidth=2,
                    label=f"LR {label} (AUC={auc:.3f}, 95% CI: {lo:.3f}-{hi:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("1 - specificity", color=INK2)
        ax.set_ylabel("Sensitivity", color=INK2)
        ax.set_title(f"({panel}) {cohort}", color=INK, fontsize=12)
        ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=170, facecolor=SURFACE)
    pd.DataFrame(rows).to_csv(out / "roc_auc.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("Saved roc_curves.png")


if __name__ == "__main__":
    main()
