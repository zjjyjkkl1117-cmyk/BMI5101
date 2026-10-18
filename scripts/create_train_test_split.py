import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# =====================================================
# 1. Configuration
# =====================================================

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.30

ID_COL = "subject_id"
TARGET_COL = "died_in_hospital"

# =====================================================
# 2. Load Model A and Model B
# =====================================================

model_a = pd.read_csv(
    DATA_DIR / "Model A (replication).csv"
)

model_b = pd.read_csv(
    DATA_DIR / "Model B (ours).csv"
)

print("Model A shape:", model_a.shape)
print("Model B shape:", model_b.shape)

# =====================================================
# 3. Validate patient IDs and outcomes
# =====================================================

for name, df in [("Model A", model_a),
                 ("Model B", model_b)]:

    assert ID_COL in df.columns, f"{name}: missing subject_id"
    assert TARGET_COL in df.columns, f"{name}: missing outcome"

    assert df[ID_COL].notna().all(), f"{name}: missing patient IDs"
    assert df[ID_COL].is_unique, f"{name}: duplicate patients"

    assert df[TARGET_COL].notna().all(), f"{name}: missing outcomes"
    assert df[TARGET_COL].isin([0, 1]).all(), f"{name}: invalid outcomes"

# Both datasets must contain exactly the same patients
assert set(model_a[ID_COL]) == set(model_b[ID_COL]), (
    "Model A and Model B have different patients!"
)

# Check outcome labels match for every patient
labels_a = model_a.set_index(ID_COL)[TARGET_COL].sort_index()
labels_b = model_b.set_index(ID_COL)[TARGET_COL].sort_index()

assert labels_a.equals(labels_b), (
    "Outcome labels differ between Model A and Model B!"
)

print("\nPatient IDs and outcomes are consistent.")

# =====================================================
# 4. Create one shared Train/Test split
# =====================================================

patients = model_b[
    [ID_COL, TARGET_COL]
].sort_values(ID_COL).reset_index(drop=True)

train_ids, test_ids = train_test_split(
    patients[ID_COL],
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=patients[TARGET_COL]
)

train_ids = set(train_ids)
test_ids = set(test_ids)

# =====================================================
# 5. Apply identical split to Model A and Model B
# =====================================================

A_train = model_a[
    model_a[ID_COL].isin(train_ids)
].copy()

A_test = model_a[
    model_a[ID_COL].isin(test_ids)
].copy()

B_train = model_b[
    model_b[ID_COL].isin(train_ids)
].copy()

B_test = model_b[
    model_b[ID_COL].isin(test_ids)
].copy()

# Sort by patient ID for consistent row ordering
A_train = A_train.sort_values(ID_COL)
A_test = A_test.sort_values(ID_COL)

B_train = B_train.sort_values(ID_COL)
B_test = B_test.sort_values(ID_COL)

# =====================================================
# 6. Verify split consistency
# =====================================================

assert set(A_train[ID_COL]) == set(B_train[ID_COL])
assert set(A_test[ID_COL]) == set(B_test[ID_COL])

assert set(A_train[ID_COL]).isdisjoint(
    set(A_test[ID_COL])
)

assert len(A_train) + len(A_test) == len(model_a)
assert len(B_train) + len(B_test) == len(model_b)

print("\nAll split checks passed!")

# =====================================================
# 7. Save the four datasets
# =====================================================

A_train.to_csv(OUTPUT_DIR / "A_train.csv", index=False)
A_test.to_csv(OUTPUT_DIR / "A_test.csv", index=False)

B_train.to_csv(OUTPUT_DIR / "B_train.csv", index=False)
B_test.to_csv(OUTPUT_DIR / "B_test.csv", index=False)

# =====================================================
# 8. Save shared patient split
# =====================================================

split_df = patients[[ID_COL]].copy()

split_df["split"] = split_df[ID_COL].apply(
    lambda x: "train" if x in train_ids else "test"
)

split_df.to_csv(
    OUTPUT_DIR / "train_test_split.csv",
    index=False
)

# =====================================================
# 9. Print results
# =====================================================

print("\n========== SPLIT SUMMARY ==========")

print("Random seed:", RANDOM_SEED)

print("\nModel A:")
print("Training:", A_train.shape)
print("Testing:", A_test.shape)

print("\nModel B:")
print("Training:", B_train.shape)
print("Testing:", B_test.shape)

print("\nMortality rates:")
print("A train:", A_train[TARGET_COL].mean())
print("A test:", A_test[TARGET_COL].mean())
print("B train:", B_train[TARGET_COL].mean())
print("B test:", B_test[TARGET_COL].mean())

print("\nFiles saved to:", OUTPUT_DIR)
