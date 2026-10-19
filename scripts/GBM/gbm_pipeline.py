"""P15 GBM - candidate features, fold-safe preprocessing, LASSO selection and GBM tuning.

Imported by gbm_cv.py. Kept as its own module so the saved .joblib pipelines can be
loaded from any script: classes pickled from a script's __main__ cannot.

Preprocessing and LASSO follow scripts/xgboost_cv.py (median imputation with missing
indicators, standard scaling, one-hot encoding, L1 logistic regression via
SelectFromModel), with two additions agreed for this analysis:
  * LASSO strength C is tuned by inner cross-validation instead of fixed at 0.5.
  * Ventilator-hour columns get structural zeros before imputation
    (missing = not ventilated -> 0 h; negative values -> 0 h).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ID = "subject_id"
TARGET = "died_in_hospital"

# --------------------------------------------------------------------------- #
# Candidate features: the paper-matched pools in model_{A,B}_selected_features.csv
# (every row of those files, selected or not). Already filtered to <=30% missing
# in the training set by the earlier LASSO pipeline (a label-free rule).
# --------------------------------------------------------------------------- #
_DEMOGRAPHIC = ["age_at_admission", "female", "race_white", "weight_admit_kg"]
_COMORBIDITY = ["prior_chf", "prior_ckd", "prior_copd", "prior_diabetes",
                "prior_liver_disease"]
_SCORES = ["apsiii", "gcs_total_min", "sapsii", "sofa_day1"]
_VITALS = ["mean_dbp", "mean_heart_rate", "mean_map", "mean_resp_rate", "mean_sbp",
           "mean_spo2", "mean_temp_c"]
_LABS = ["mean_anion_gap", "mean_bicarbonate", "mean_bun", "mean_chloride",
         "mean_creatinine", "mean_glucose", "mean_hemoglobin", "mean_inr",
         "mean_platelet", "mean_potassium", "mean_pt", "mean_ptt", "mean_rdw",
         "mean_sodium", "mean_wbc"]
_TREATMENTS = ["crrt_by_t0", "invasive_vent_by_t0", "losartan_by_t0", "on_dopamine",
               "on_epinephrine", "on_norepinephrine", "on_vasopressin"]

# Known only after discharge / extubation. Allowed in Model A (the replication) only.
LEAKY_MODEL_A_ONLY = ["charlson_comorbidity_index", "los_hospital_days", "ventilator_hours"]

MODEL_A_FEATURES = (_DEMOGRAPHIC + _COMORBIDITY + ["charlson_comorbidity_index"]
                    + _SCORES + _VITALS + _LABS + _TREATMENTS
                    + ["ventilator_hours", "los_hospital_days"])
MODEL_B_FEATURES = (_DEMOGRAPHIC + _COMORBIDITY + _SCORES + _VITALS + _LABS
                    + _TREATMENTS + ["vent_hours_within_24h"])
SENSITIVITY_DROPPED = ["sapsii", "apsiii"]

VARIANTS = {
    "A": {"data": "A", "features": MODEL_A_FEATURES,
          "label": "Model A (replication; contains leakage)"},
    "B": {"data": "B", "features": MODEL_B_FEATURES,
          "label": "Model B (leakage-reduced)"},
    "B_noscores": {"data": "B",
                   "features": [f for f in MODEL_B_FEATURES if f not in SENSITIVITY_DROPPED],
                   "label": "Model B without SAPS II and APS III (sensitivity)"},
}
ARMS = {"lasso": "GBM + LASSO", "all": "GBM, all eligible features"}

# Timing audit, one line per candidate (written to the data audit).
TIMING_NOTES = {
    "charlson_comorbidity_index": "LEAKAGE: discharge-coded diagnoses (Model A only)",
    "los_hospital_days": "LEAKAGE: whole hospital stay (Model A only)",
    "ventilator_hours": "LEAKAGE: whole-stay ventilation (Model A only); missing -> 0 h",
    "sapsii": "FLAG: metastatic cancer / haematologic malignancy / AIDS points from "
              "discharge ICD codes (mimic-code sapsii)",
    "apsiii": "FLAG: worst PaO2 / A-a gradient / acid-base over the whole ICU stay; "
              "CKD from discharge ICD codes (mimic-code apsiii)",
    "vent_hours_within_24h": "first 24 h, but sums every derived.ventilation row incl. "
                             "supplemental oxygen; missing/negative -> 0 h",
    "sofa_day1": "ICU intime - 6 h to + 24 h (first_day_sofa)",
    "gcs_total_min": "minimum in [intime, t0]",
    "weight_admit_kg": "charted weight up to t0",
    "race_white": "admission record",
    "age_at_admission": "admission record",
    "female": "patient record",
    "crrt_by_t0": "CRRT charted up to t0",
    "losartan_by_t0": "prescription started up to t0",
    "invasive_vent_by_t0": "invasive ventilation started up to t0",
}
for _c in _COMORBIDITY:
    TIMING_NOTES[_c] = "earlier admissions only (weaker proxy than the paper's)"
for _c in _VITALS + _LABS:
    TIMING_NOTES[_c] = "mean of values charted in [intime, t0]"
for _c in ["on_dopamine", "on_epinephrine", "on_norepinephrine", "on_vasopressin"]:
    TIMING_NOTES[_c] = "infusion started in [intime, t0]"

# Columns that must never be predictors in any variant.
FORBIDDEN = re.compile(
    r"^(subject_id|hadm_id|first_stay_id|stay_ids?|paper_would_exclude|"
    r"died_.*|hospital_expire_flag|cmo_before_t0|dnr_before_t0)$"
    r"|los_icu|icu_los|outtime|dischtime|deathtime|discharge_location|hospice"
)

STRUCTURAL_ZERO = {
    "ventilator_hours": "missing = never ventilated -> 0 h",
    "vent_hours_within_24h": "missing = not ventilated in first 24 h -> 0 h; "
                             "negative (ventilation began before ICU) -> 0 h",
}

# --------------------------------------------------------------------------- #
# Tuning grids
# --------------------------------------------------------------------------- #
INNER_FOLDS = 5
LASSO_CS = np.logspace(-3, 1, 30)          # inverse regularisation strength
GBM_N_ITER = 30                            # random-search settings per tuning run
GBM_SPACE = {
    "gbm__n_estimators": [100, 200, 300, 500],
    "gbm__learning_rate": [0.01, 0.02, 0.05, 0.1],
    "gbm__max_depth": [2, 3, 4],
    "gbm__min_samples_leaf": [5, 10, 20, 40],
    "gbm__subsample": [0.6, 0.8, 1.0],
    "gbm__max_features": [None, "sqrt", 0.5],
}
SCORING = "neg_log_loss"   # proper scoring rule: rewards ranking and calibration


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
class StructuralZeros(BaseEstimator, TransformerMixin):
    """Stateless: missing -> 0 and negative -> 0 for the listed columns.

    Uses no outcome information and learns nothing from the data, so it is
    identical whether applied inside or outside a CV fold."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            if c in X.columns:
                X[c] = X[c].fillna(0).clip(lower=0)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def _sk_version():
    major, minor = (int(v) for v in re.findall(r"\d+", sklearn.__version__)[:2])
    return major, minor


def l1_logistic(C=1.0):
    """L1 logistic regression on every scikit-learn version.

    `penalty=` is deprecated from 1.8; before 1.8, l1_ratio alone would silently
    give an L2 model, so the branch matters."""
    kw = dict(solver="liblinear", C=C, max_iter=5000, random_state=SEED)
    if _sk_version() >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, **kw)
    return LogisticRegression(penalty="l1", **kw)


def make_preprocess(x: pd.DataFrame) -> ColumnTransformer:
    """Same preprocessing as xgboost_cv.py."""
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return ColumnTransformer([
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


def _front(x):
    zeros = [c for c in STRUCTURAL_ZERO if c in x.columns]
    return [("zeros", StructuralZeros(columns=zeros)), ("preprocess", make_preprocess(x))]


def build_lasso_logistic(x: pd.DataFrame, C=1.0) -> Pipeline:
    return Pipeline(_front(x) + [("l1", l1_logistic(C))])


def build_pipeline(x: pd.DataFrame, arm: str, C=None, gbm_params=None) -> Pipeline:
    steps = _front(x)
    if arm == "lasso":
        steps.append(("lasso", SelectFromModel(l1_logistic(C), threshold=1e-8)))
    elif arm != "all":
        raise ValueError(arm)
    steps.append(("gbm", GradientBoostingClassifier(random_state=SEED,
                                                     **(gbm_params or {}))))
    return Pipeline(steps)


def clean_name(n: str) -> str:
    return re.sub(r"^(numeric|categorical)__", "", n)


def output_feature_names(pipe: Pipeline) -> list[str]:
    names = [clean_name(n) for n in pipe.named_steps["preprocess"].get_feature_names_out()]
    if "lasso" in pipe.named_steps:
        mask = pipe.named_steps["lasso"].get_support()
        names = [n for n, keep in zip(names, mask) if keep]
    return names


# --------------------------------------------------------------------------- #
# Tuning (always on training rows only; cv splits those rows)
# --------------------------------------------------------------------------- #
def tune_lasso_C(x, y, cv, n_jobs=1, Cs=LASSO_CS):
    """Choose C by cross-validated log loss of the L1 logistic model (lambda.min).

    Preprocessing is refitted inside every inner fold."""
    search = GridSearchCV(build_lasso_logistic(x), {"l1__C": list(Cs)},
                          scoring=SCORING, cv=cv, n_jobs=n_jobs, refit=False)
    search.fit(x, y)
    table = pd.DataFrame({"C": search.cv_results_["param_l1__C"].astype(float),
                          "cv_log_loss": -search.cv_results_["mean_test_score"],
                          "cv_log_loss_sd": search.cv_results_["std_test_score"]})
    return float(search.best_params_["l1__C"]), table


def fit_tuned(x, y, arm, cv, n_jobs=1, n_iter=GBM_N_ITER, Cs=LASSO_CS):
    """Tune (LASSO C, then GBM) on (x, y) with `cv`, refit on all of (x, y).

    Returns the fitted pipeline and a dict describing the choices."""
    info = {}
    C = None
    if arm == "lasso":
        C, _ = tune_lasso_C(x, y, cv, n_jobs=n_jobs, Cs=Cs)
        info["lasso_C"] = C
    search = RandomizedSearchCV(build_pipeline(x, arm, C=C), GBM_SPACE, n_iter=n_iter,
                                scoring=SCORING, cv=cv, random_state=SEED,
                                n_jobs=n_jobs, refit=True)
    search.fit(x, y)
    info["gbm_params"] = {k.replace("gbm__", ""): v
                          for k, v in search.best_params_.items()}
    info["inner_cv_log_loss"] = float(-search.best_score_)
    pipe = search.best_estimator_
    chosen = output_feature_names(pipe)
    if not chosen:
        raise RuntimeError("LASSO selected zero features")
    info["n_features"] = len(chosen)
    info["features"] = chosen
    return pipe, info
