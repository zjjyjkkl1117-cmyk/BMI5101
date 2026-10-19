"""P15: build the Model A / Model B modelling files and the shared train/test split.

Inputs (project/data/raw/):
    new_model_A.csv, new_model_B.csv                  BigQuery exports (all columns)
    model_A_selected_features.csv, model_B_selected_features.csv   LASSO-list outputs

Outputs:
    data/Model A (replication).csv, data/Model B (ours).csv
        subject_id, hadm_id, first_stay_id, the candidate features, died_in_hospital
    splits/train_test_split.csv   fixed patient-level assignment (no longer depends on row order)
    splits/train_patient_ids.csv  same training IDs, one column
    splits/{A,B}_{train,test}.csv same layout as create_train_test_split.py writes

The split is the one the LASSO lists were built on: sklearn train_test_split
(test_size=0.30, random_state=42, stratified on died_in_hospital) applied to the rows of
new_model_A.csv in their file order. create_train_test_split.py sorts by subject_id first
and therefore gives a different split. The script checks that this split reproduces the
training-set missingness reported in both selected-feature files.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gbm_pipeline as gp  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
ID, TARGET, SEED, TEST_SIZE = "subject_id", "died_in_hospital", 42, 0.30
KEEP_IDS = ["subject_id", "hadm_id", "first_stay_id"]


def main():
    raw_a = pd.read_csv(RAW / "new_model_A.csv")
    raw_b = pd.read_csv(RAW / "new_model_B.csv")
    lists = {"A": pd.read_csv(RAW / "model_A_selected_features.csv"),
             "B": pd.read_csv(RAW / "model_B_selected_features.csv")}

    for name, df in [("A", raw_a), ("B", raw_b)]:
        assert df[ID].notna().all() and df[ID].is_unique, f"{name}: bad subject_id"
        assert df[TARGET].isin([0, 1]).all(), f"{name}: bad outcome"
    a_y = raw_a.set_index(ID)[TARGET].sort_index()
    b_y = raw_b.set_index(ID)[TARGET].sort_index()
    assert a_y.index.equals(b_y.index) and a_y.equals(b_y), "A/B patients or outcomes differ"

    # candidate pools must equal the selected-feature files exactly
    for name, feats in [("A", gp.MODEL_A_FEATURES), ("B", gp.MODEL_B_FEATURES)]:
        assert set(feats) == set(lists[name]["feature"]), f"{name}: candidate list mismatch"
    assert not set(gp.LEAKY_MODEL_A_ONLY) & set(raw_b.columns), "Model B export has leaky columns"

    # split on Model A's file row order (as the LASSO lists were)
    train_ids, test_ids = train_test_split(raw_a[ID], test_size=TEST_SIZE,
                                           random_state=SEED, stratify=raw_a[TARGET])
    train_ids, test_ids = set(train_ids), set(test_ids)

    # verify: training missingness reproduces both lists (structural-zero columns excluded)
    for name, raw in [("A", raw_a), ("B", raw_b)]:
        lst = lists[name].set_index("feature")
        feats = [f for f in lst.index if f not in gp.STRUCTURAL_ZERO]
        tr = raw[raw[ID].isin(train_ids)]
        ours = (100 * tr[feats].isna().mean()).round(1)
        bad = ours[(ours - lst.loc[feats, "train_missing_pct"]).abs() > 0.05]
        assert bad.empty, f"{name}: split does not reproduce the LASSO list ({list(bad.index)})"
        print(f"Model {name}: training missingness matches the LASSO list for "
              f"{len(feats)}/{len(feats)} features")

    out_data = PROJECT / "data"
    out_split = PROJECT / "splits"
    out_split.mkdir(parents=True, exist_ok=True)
    order = raw_a[ID]
    files = {}
    for name, raw, feats, fname in [
            ("A", raw_a, gp.MODEL_A_FEATURES, "Model A (replication).csv"),
            ("B", raw_b, gp.MODEL_B_FEATURES, "Model B (ours).csv")]:
        cols = [c for c in KEEP_IDS + feats + [TARGET] if c != ID]
        df = raw.set_index(ID).loc[order, cols].reset_index()
        df.to_csv(out_data / fname, index=False)
        files[name] = df
        print(f"{fname}: {df.shape[0]} rows, {len(feats)} candidate features")

    manifest = pd.DataFrame({ID: sorted(order)})
    manifest["split"] = np.where(manifest[ID].isin(train_ids), "train", "test")
    manifest.to_csv(out_split / "train_test_split.csv", index=False)
    pd.DataFrame({ID: sorted(train_ids)}).to_csv(out_split / "train_patient_ids.csv",
                                                 index=False)
    for name, df in files.items():
        for part, ids in [("train", train_ids), ("test", test_ids)]:
            df[df[ID].isin(ids)].sort_values(ID).to_csv(
                out_split / f"{name}_{part}.csv", index=False)
    assert not train_ids & test_ids
    y = files["A"].set_index(ID)[TARGET]
    print(f"Split: train {len(train_ids)} ({int(y[list(train_ids)].sum())} deaths), "
          f"test {len(test_ids)} ({int(y[list(test_ids)].sum())} deaths), seed {SEED}")


if __name__ == "__main__":
    main()
