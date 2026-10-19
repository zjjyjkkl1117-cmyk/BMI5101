"""
metrics.py  --  P15 team standard for evaluating ICU lung-cancer mortality models
================================================================================

Every model in the project (LR, SVM, GBM, ANN, XGBoost, KNN, AdaBoost, RF ...)
should be evaluated with THIS file so the numbers are comparable.

Team rules (fixed here so nobody has to decide them again)
---------------------------------------------------------
1. Threshold: Youden's J (max sensitivity + specificity - 1), chosen on the
   10-fold OUT-OF-FOLD predicted probabilities of the TRAINING set, then applied
   unchanged to the training and test sets. The test set never chooses anything.
2. AUROC 95% CI: percentile bootstrap, 2,000 resamples, seed 42.
3. Metrics: AUROC (+CI), accuracy, F1, sensitivity, specificity, plus PPV, NPV
   and Brier score (calibration).
4. Three rows per model:
     Train (apparent)   model fit on all training data, predicting training data
                        (this is what the paper reports as "training cohort")
     Train (10-fold CV) out-of-fold predictions on the training data
     Test               model fit on all training data, predicting the test set
                        (the honest number to compare against the paper)

Typical use (any scikit-learn-style estimator, including xgboost.XGBClassifier)
-------------------------------------------------------------------------------
    from metrics import oof_proba, evaluate_model

    model  = XGBClassifier(**best_params)                    # already tuned
    p_oof  = oof_proba(model, X_train, y_train)               # 1) threshold source
    model.fit(X_train, y_train)                               # 2) final fit
    p_tr   = model.predict_proba(X_train)[:, 1]
    p_te   = model.predict_proba(X_test)[:, 1]
    table  = evaluate_model("XGBoost", y_train, p_tr, y_test, p_te, p_oof)
    table.to_csv("xgboost_performance.csv", index=False)

If you undersample / oversample, do it ONLY inside training (inside each CV
fold for oof_proba too, e.g. with imblearn's Pipeline), and still pass
probabilities for the full, original training and test sets to evaluate_model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RANDOM_STATE = 42
N_FOLDS = 10
N_BOOT = 2000

__all__ = [
    "youden_threshold",
    "auroc_ci",
    "classification_metrics",
    "oof_proba",
    "evaluate_model",
    "format_table",
]


def _as_1d(a) -> np.ndarray:
    return np.asarray(a).ravel()


def youden_threshold(y_true, y_prob) -> float:
    """Probability cut-off that maximises Youden's J = sensitivity + specificity - 1."""
    y_true, y_prob = _as_1d(y_true), _as_1d(y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    finite = np.isfinite(thr)                       # first threshold is +inf
    j = tpr[finite] - fpr[finite]
    return float(np.clip(thr[finite][int(np.argmax(j))], 0.0, 1.0))


def auroc_ci(y_true, y_prob, n_boot: int = N_BOOT, alpha: float = 0.05,
             seed: int = RANDOM_STATE) -> tuple[float, float, float]:
    """AUROC with a percentile-bootstrap (1 - alpha) confidence interval."""
    y_true, y_prob = _as_1d(y_true), _as_1d(y_prob)
    auc = roc_auc_score(y_true, y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():  # resample with one class only
            continue
        boots.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(auc), float(lo), float(hi)


def classification_metrics(y_true, y_prob, threshold: float, n_boot: int = N_BOOT,
                           seed: int = RANDOM_STATE) -> dict:
    """All team metrics at one fixed threshold (predicted positive if prob >= threshold)."""
    y_true, y_prob = _as_1d(y_true).astype(int), _as_1d(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_div(a, b):
        return float(a / b) if b else float("nan")

    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    auc, lo, hi = auroc_ci(y_true, y_prob, n_boot=n_boot, seed=seed)
    return {
        "N": int(len(y_true)),
        "Deaths": int(y_true.sum()),
        "AUROC": auc,
        "AUROC_CI_low": lo,
        "AUROC_CI_high": hi,
        "Accuracy": safe_div(tp + tn, len(y_true)),
        "F1": f1,
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Threshold": float(threshold),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def oof_proba(estimator, X, y, n_folds: int = N_FOLDS, seed: int = RANDOM_STATE) -> np.ndarray:
    """Out-of-fold P(death) for every training patient (stratified k-fold, fixed seed).

    `estimator` is cloned, so pass the model with its final (tuned) hyper-parameters;
    it does not need to be fitted.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(clone(estimator), X, _as_1d(y), cv=cv, method="predict_proba")
    return proba[:, 1]


def evaluate_model(model_name: str, y_train, p_train, y_test, p_test, p_train_oof,
                   threshold: float | None = None, n_boot: int = N_BOOT,
                   seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Team-standard performance table (3 rows) for one model.

    p_train      probabilities on the training set from the model fit on all training data
    p_test       probabilities on the test set from the same model
    p_train_oof  out-of-fold training probabilities (from oof_proba); used for the threshold
    threshold    leave as None to use the team rule (Youden on p_train_oof)
    """
    if threshold is None:
        threshold = youden_threshold(y_train, p_train_oof)
    rows = []
    for cohort, y, p in [("Train (apparent)", y_train, p_train),
                         ("Train (10-fold CV)", y_train, p_train_oof),
                         ("Test", y_test, p_test)]:
        rows.append({"Model": model_name, "Cohort": cohort,
                     **classification_metrics(y, p, threshold, n_boot=n_boot, seed=seed)})
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Presentation version: 'AUROC (95% CI)' column, rounded, confusion counts dropped."""
    out = df.copy()
    f = f"{{:.{digits}f}}"
    out["AUROC (95% CI)"] = [f"{f.format(a)} ({f.format(l)}-{f.format(h)})"
                             for a, l, h in zip(out.AUROC, out.AUROC_CI_low, out.AUROC_CI_high)]
    cols = ["Model", "Cohort", "N", "Deaths", "AUROC (95% CI)", "Accuracy", "F1",
            "Sensitivity", "Specificity", "PPV", "NPV", "Brier", "Threshold"]
    out = out[cols]
    num = ["Accuracy", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "Brier", "Threshold"]
    out[num] = out[num].astype(float).round(digits)
    return out
