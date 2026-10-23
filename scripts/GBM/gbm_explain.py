"""P15: GBM feature importance and SHAP beeswarm plots for the frozen final models.

Usage (from the project root, after gbm_cv.py --final-test):
    python scripts/gbm_explain.py                 # SHAP on the held-out test set (default)
    python scripts/gbm_explain.py --data train    # SHAP on the 994 training patients

Explains the six frozen pipelines in results/gbm/models/ without refitting anything.

* GBM importance  = scikit-learn's impurity-based feature_importances_ (share of the total
  loss reduction contributed by splits on each feature, summing to 100%).
* SHAP            = exact TreeSHAP (shap.TreeExplainer) on the fitted GBM. Values are on the
  log-odds scale; for every patient, base value + sum of SHAP values equals the model's
  log-odds output (checked). Positive SHAP = pushes predicted mortality up.
* Beeswarm colour = the patient's ORIGINAL feature value (before imputation/scaling),
  scaled within each feature to its 5th-95th percentile; grey = value missing (imputed).
  Styled after shapviz::sv_importance(kind = "beeswarm"): inferno palette 0.25-0.85,
  features ordered by mean |SHAP|, top 15 shown.

Outputs: results/gbm/explain/{figures,tables}/
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import gbm_pipeline as gp  # noqa: E402

TOP_N = 15
INK, INK_2, GRID, FRAME, MISSING_GREY = "#1a1a1a", "#4d4d4d", "#ebebeb", "#333333", "#bdbdbd"
BAR_OK, BAR_LEAK = "#2a78d6", "#eb6834"
CMAP = LinearSegmentedColormap.from_list(
    "inferno_25_85", plt.get_cmap("inferno")(np.linspace(0.25, 0.85, 256)))

LABELS = {
    "age_at_admission": "Age", "female": "Female sex", "race_white": "Race: White",
    "weight_admit_kg": "Weight", "prior_chf": "Heart failure (prior adm.)",
    "prior_ckd": "Renal disease (prior adm.)", "prior_copd": "COPD (prior adm.)",
    "prior_diabetes": "Diabetes (prior adm.)", "prior_liver_disease": "Liver disease (prior adm.)",
    "charlson_comorbidity_index": "Charlson index", "apsiii": "APS III",
    "gcs_total_min": "GCS (minimum)", "sapsii": "SAPS II", "sofa_day1": "SOFA",
    "mean_dbp": "Diastolic BP", "mean_heart_rate": "Heart rate", "mean_map": "Mean arterial BP",
    "mean_resp_rate": "Respiratory rate", "mean_sbp": "Systolic BP", "mean_spo2": "SpO₂",
    "mean_temp_c": "Temperature", "mean_anion_gap": "Anion gap",
    "mean_bicarbonate": "Bicarbonate", "mean_bun": "BUN", "mean_chloride": "Chloride",
    "mean_creatinine": "Creatinine", "mean_glucose": "Glucose", "mean_hemoglobin": "Hemoglobin",
    "mean_inr": "INR", "mean_platelet": "Platelets", "mean_potassium": "Potassium",
    "mean_pt": "PT", "mean_ptt": "PTT", "mean_rdw": "RDW", "mean_sodium": "Sodium",
    "mean_wbc": "WBC", "crrt_by_t0": "CRRT", "invasive_vent_by_t0": "Invasive ventilation",
    "losartan_by_t0": "Losartan", "on_dopamine": "Dopamine", "on_epinephrine": "Epinephrine",
    "on_norepinephrine": "Norepinephrine", "on_vasopressin": "Vasopressin",
    "ventilator_hours": "Ventilator hours (whole stay)",
    "vent_hours_within_24h": "Ventilation hours (first 24 h)",
    "los_hospital_days": "Hospital length of stay",
}
LEAK_MARK = "†"     # dagger after the label of a leakage variable


def label(feature: str) -> str:
    base = feature.replace("missingindicator_", "")
    text = LABELS.get(base, base)
    if feature.startswith("missingindicator_"):
        text += " missing"
    return text + (f" {LEAK_MARK}" if base in gp.LEAKY_MODEL_A_ONLY else "")


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11.5, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def framed(ax):
    """ggplot theme_bw look: white panel, light grid, thin dark frame."""
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(FRAME)
        s.set_linewidth(0.8)
    ax.tick_params(length=3, color=FRAME)


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #
def explain(pipe, x_raw):
    """SHAP matrix, transformed names, original-unit colour values, base value."""
    names = gp.output_feature_names(pipe)
    xt = pipe[:-1].transform(x_raw)
    gbm = pipe.named_steps["gbm"]
    ex = shap.TreeExplainer(gbm)
    sv = np.asarray(ex.shap_values(xt))
    base = float(np.ravel(ex.expected_value)[0])
    err = np.abs(sv.sum(1) + base - gbm.decision_function(xt)).max()
    assert err < 1e-8, f"SHAP additivity failed ({err})"
    z = pipe.named_steps["zeros"].transform(x_raw)     # structural zeros only, no imputation
    colour = pd.DataFrame({
        n: (z[n.replace("missingindicator_", "")].isna().astype(float)
            if n.startswith("missingindicator_") else z[n].astype(float))
        for n in names}, index=x_raw.index)
    return sv, names, colour, base, gbm.feature_importances_, err


def summary_table(key, sv, names, colour, importance):
    rows = []
    for j, n in enumerate(names):
        c, s = colour[n].to_numpy(), sv[:, j]
        ok = ~np.isnan(c)
        rho = (spearmanr(c[ok], s[ok]).statistic
               if ok.sum() > 2 and np.unique(c[ok]).size > 1 and np.unique(s[ok]).size > 1
               else np.nan)       # all-zero SHAP: the trees never split on this feature
        rows.append({"model": key, "feature": n, "label": label(n),
                     "mean_abs_shap": np.abs(s).mean(), "gbm_importance_pct": 100 * importance[j],
                     "spearman_value_vs_shap": rho,
                     "direction": ("higher value -> higher risk" if rho > 0.1 else
                                   "higher value -> lower risk" if rho < -0.1 else
                                   "mixed / non-monotone"),
                     "pct_missing": 100 * float(np.isnan(c).mean()),
                     "leakage": n.replace("missingindicator_", "") in gp.LEAKY_MODEL_A_ONLY})
    t = pd.DataFrame(rows)
    t["rank_shap"] = t["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    t["rank_gbm_importance"] = t["gbm_importance_pct"].rank(ascending=False,
                                                            method="first").astype(int)
    return t.sort_values("rank_shap")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def beeswarm_offsets(s, row_height=0.40, nbins=100, seed=0):
    """Vertical offsets that stack points sharing an x-bin (as shap's beeswarm)."""
    rng = np.random.default_rng(seed)
    span = s.max() - s.min()
    quant = np.round(nbins * (s - s.min()) / (span + 1e-8))
    order = np.argsort(quant + rng.random(len(s)) * 1e-6)
    ys, layer, last = np.zeros(len(s)), 0, None
    for i in order:
        if quant[i] != last:
            layer = 0
        ys[i] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last = quant[i]
    return ys * (0.9 * row_height / (np.abs(ys).max() + 1))


def colour_scale(c):
    ok = ~np.isnan(c)
    lo, hi = np.nanpercentile(c, [5, 95]) if ok.any() else (0, 1)
    if hi <= lo:
        lo, hi = np.nanmin(c), np.nanmax(c)
    if hi <= lo:
        hi = lo + 1
    return np.clip((c - lo) / (hi - lo), 0, 1)


def draw_beeswarm(ax, sv, names, colour, top=TOP_N):
    order = np.argsort(-np.abs(sv).mean(0))[:top]
    has_missing = False
    for row, j in enumerate(order[::-1]):              # most important at the top
        s, c = sv[:, j], colour[names[j]].to_numpy()
        y = row + beeswarm_offsets(s, seed=row)
        miss = np.isnan(c)
        has_missing |= bool(miss.any())
        ax.scatter(s[miss], y[miss], s=13, color=MISSING_GREY, linewidths=0, zorder=2)
        k = np.flatnonzero(~miss)
        k = k[np.random.default_rng(row).permutation(len(k))]
        ax.scatter(s[k], y[k], s=13, c=colour_scale(c)[k], cmap=CMAP, vmin=0, vmax=1,
                   linewidths=0, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label(names[j]) for j in order[::-1]])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("SHAP value (log-odds of in-hospital death)")
    framed(ax)
    return has_missing


def colourbar(fig, ax, has_missing):
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03, aspect=22, shrink=0.55)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.ax.tick_params(length=0, labelsize=10.5, colors=INK)
    cb.outline.set_visible(False)
    cb.set_label("Feature value", fontsize=11.5, color=INK, labelpad=-2)
    if has_missing:
        cb.ax.text(0.5, -0.12, "grey =\nmissing", transform=cb.ax.transAxes, ha="center",
                   va="top", fontsize=8.5, color=INK_2)


def draw_importance(ax, table, top=TOP_N):
    t = table.sort_values("gbm_importance_pct", ascending=False).head(top)[::-1]
    colours = [BAR_LEAK if leak else BAR_OK for leak in t["leakage"]]
    ax.barh(range(len(t)), t["gbm_importance_pct"], height=0.62, color=colours, zorder=3)
    xmax = t["gbm_importance_pct"].max()
    for i, v in enumerate(t["gbm_importance_pct"]):
        ax.text(v + xmax * 0.015, i, f"{v:.1f}", va="center", ha="left", fontsize=9.5,
                color=INK_2)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["label"])
    ax.set_xlim(0, xmax * 1.14)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.set_xlabel("GBM feature importance (% of total loss reduction)")
    framed(ax)
    ax.grid(False, axis="y")
    return bool(t["leakage"].any())


def caption(ax, text):
    """Small grey note just below the x-axis title, aligned with the panel's left edge."""
    ax.annotate(text, xy=(0, 0), xycoords="axes fraction", xytext=(0, -46),
                textcoords="offset points", ha="left", va="top", fontsize=8.5, color=INK_2)


def save(fig, stem):
    fig.savefig(f"{stem}.png")
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["test", "train"], default="test")
    ap.add_argument("--models", default="A_lasso,A_all,B_lasso,B_all,B_noscores_lasso,B_noscores_all")
    ap.add_argument("--results", type=Path, default=SCRIPTS.parent / "results" / "gbm")
    ap.add_argument("--splits", type=Path, default=SCRIPTS.parent / "splits")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    out = args.results / "explain"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    set_style()
    who = ("held-out test set" if args.data == "test" else "training set")
    arm_title = {"lasso": "GBM + LASSO", "all": "GBM, all features"}
    model_title = {"A": "Model A (replication)", "B": "Model B (leakage-reduced)",
                   "B_noscores": "Model B without SAPS II / APS III"}

    tables, cache = [], {}
    for key in [k.strip() for k in args.models.split(",")]:
        v, arm = key.rsplit("_", 1)
        df = pd.read_csv(args.splits / f"{gp.VARIANTS[v]['data']}_{args.data}.csv")
        pipe = joblib.load(args.results / "models" / f"{key}_final_pipeline.joblib")
        x = df[gp.VARIANTS[v]["features"]]
        sv, names, colour, base, imp, err = explain(pipe, x)
        t = summary_table(key, sv, names, colour, imp)
        tables.append(t)
        cache[key] = (sv, names, colour, t)
        n = len(df)
        print(f"{key:18s} n={n} features={len(names)} base log-odds={base:.3f} "
              f"additivity error={err:.1e} | top SHAP: "
              + ", ".join(t["label"].head(5)), flush=True)

        title = f"{arm_title[arm]} — {model_title[v]}"
        foot_leak = (f"   {LEAK_MARK} not available at 24 h (leakage; Model A only)."
                     if v == "A" else "")
        # importance
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        leak = draw_importance(ax, t)
        ax.set_title(title, pad=10)
        if leak:
            ax.legend(handles=[Patch(color=BAR_OK, label="Available by 24 h"),
                               Patch(color=BAR_LEAK, label=f"Leakage {LEAK_MARK}")],
                      loc="lower right", frameon=True, edgecolor=GRID, fontsize=9.5)
        caption(ax, f"Impurity-based importance of the fitted GBM (top {TOP_N} of "
                     f"{len(names)} inputs; all inputs sum to 100%). Vitals and labs = "
                     f"mean over the first 24 h.{foot_leak}")
        save(fig, out / "figures" / f"gbm_importance_{key}")
        # beeswarm
        fig, ax = plt.subplots(figsize=(7.8, 6.6))
        has_missing = draw_beeswarm(ax, sv, names, colour)
        ax.set_title(title, pad=10)
        colourbar(fig, ax, has_missing)
        caption(ax, f"SHAP values on the {who} (n = {n}); each dot is one patient. Right of 0 = "
                     f"higher predicted mortality. Colour = original feature value.\nTop {TOP_N} "
                     f"of {len(names)} inputs by mean |SHAP|. Vitals and labs = mean over the "
                     f"first 24 h.{foot_leak}")
        save(fig, out / "figures" / f"shap_beeswarm_{key}")

    # side-by-side slide figures: replication (A) vs leakage-reduced (B), GBM + LASSO
    if {"A_lasso", "B_lasso"} <= set(cache):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), gridspec_kw={"wspace": 0.62})
        hm = False
        for ax, key in zip(axes, ["A_lasso", "B_lasso"]):
            sv, names, colour, _ = cache[key]
            hm |= draw_beeswarm(ax, sv, names, colour)
            ax.set_title(model_title[key.rsplit("_", 1)[0]], pad=10)
        lo = min(a.get_xlim()[0] for a in axes)
        hi = max(a.get_xlim()[1] for a in axes)
        for a in axes:                                  # same SHAP scale in both panels
            a.set_xlim(lo, hi)
        colourbar(fig, axes[1], hm)
        caption(axes[0], f"GBM + LASSO, SHAP values on the {who}. Each dot is one patient; right of 0 "
                     f"= higher predicted mortality. {LEAK_MARK} not available at 24 h (leakage).")
        save(fig, out / "figures" / "shap_beeswarm_A_vs_B_lasso")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"wspace": 0.62})
        for ax, key in zip(axes, ["A_lasso", "B_lasso"]):
            draw_importance(ax, cache[key][3])
            ax.set_title(model_title[key.rsplit("_", 1)[0]], pad=10)
        axes[0].legend(handles=[Patch(color=BAR_OK, label="Available by 24 h"),
                                Patch(color=BAR_LEAK, label=f"Leakage {LEAK_MARK}")],
                       loc="lower right", frameon=True, edgecolor=GRID, fontsize=9.5)
        caption(axes[0], "GBM + LASSO, impurity-based feature importance (top 15; all inputs sum "
                     "to 100%).")
        save(fig, out / "figures" / "gbm_importance_A_vs_B_lasso")

    pd.concat(tables).to_csv(out / "tables" / f"importance_and_shap_{args.data}.csv", index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
