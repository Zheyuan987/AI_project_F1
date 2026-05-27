# preprocess.py
# Usage:
#   python preprocess.py
#
# Required files in the same folder:
#   train.csv
#   test.csv
#
# Outputs:
#   processed_data.csv
#   processed_train.csv
#   processed_test.csv
#   encoding_mapping.json
#   feature_columns.txt
#
# Notes:
#   - Driver is NOT used.
#   - sample_submission.csv is NOT required.
#   - Compound is encoded manually.
#   - Race is one-hot encoded.

import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

TARGET = "PitNextLap"
ID_COL = "id"

OUT_ALL = "processed_data.csv"
OUT_TRAIN = "processed_train.csv"
OUT_TEST = "processed_test.csv"
OUT_MAPPING = "encoding_mapping.json"
OUT_FEATURES = "feature_columns.txt"


COMPOUND_MAPPING = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
}


DROP_COLS = [
    "Driver",  # requested: do not use Driver
]


def add_features(df):
    df = df.copy()

    if "TyreLife" in df.columns and "LapNumber" in df.columns:
        df["TyreLife_x_LapNumber"] = df["TyreLife"] * df["LapNumber"]
        df["TyreLife_div_LapNumber"] = df["TyreLife"] / (df["LapNumber"] + 1)

    if "Position" in df.columns and "LapNumber" in df.columns:
        df["Position_x_LapNumber"] = df["Position"] * df["LapNumber"]

    if "Stint" in df.columns and "TyreLife" in df.columns:
        df["TyreLife_div_Stint"] = df["TyreLife"] / (df["Stint"] + 1)

    if "RaceProgress" in df.columns and "TyreLife" in df.columns:
        df["TyreLife_x_RaceProgress"] = df["TyreLife"] * df["RaceProgress"]

    if "LapTime (s)" in df.columns and "LapTime_Delta" in df.columns:
        df["LapTime_plus_Delta"] = df["LapTime (s)"] + df["LapTime_Delta"]

    return df


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if TARGET not in train.columns:
        raise ValueError(f"train.csv must contain target column: {TARGET}")
    if ID_COL not in train.columns or ID_COL not in test.columns:
        raise ValueError(f"train.csv and test.csv must contain id column: {ID_COL}")

    y = train[TARGET].copy()

    train_features = train.drop(columns=[TARGET]).copy()
    test_features = test.copy()

    # Drop unused columns
    for col in DROP_COLS:
        if col in train_features.columns:
            train_features = train_features.drop(columns=[col])
        if col in test_features.columns:
            test_features = test_features.drop(columns=[col])

    train_features["_is_train"] = 1
    test_features["_is_train"] = 0

    all_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    mapping = {
        "compound_mapping": COMPOUND_MAPPING,
        "one_hot_columns": {},
        "dropped_columns": DROP_COLS,
        "note": {
            "Driver": "not used",
            "Compound_encoded": "SOFT=0, MEDIUM=1, HARD=2, INTERMEDIATE=3, WET=4",
            "Race": "one-hot encoded",
            "sample_submission.csv": "not required by this preprocessing script",
        }
    }

    # Compound: manual numeric encoding
    if "Compound" in all_df.columns:
        all_df["Compound"] = all_df["Compound"].astype(str).str.upper().str.strip()
        all_df["Compound_encoded"] = all_df["Compound"].map(COMPOUND_MAPPING)

        if all_df["Compound_encoded"].isna().any():
            unknown_values = sorted(all_df.loc[all_df["Compound_encoded"].isna(), "Compound"].unique().tolist())
            mapping["unknown_compound_values"] = unknown_values
            all_df["Compound_encoded"] = all_df["Compound_encoded"].fillna(-1)

        all_df = all_df.drop(columns=["Compound"])

    # Convert numeric-looking strings into numeric values
    for col in all_df.columns:
        if col == ID_COL:
            continue

        if all_df[col].dtype == "object":
            converted = pd.to_numeric(all_df[col], errors="coerce")
            valid_ratio = converted.notna().mean()

            if valid_ratio > 0.95:
                all_df[col] = converted

    # Race one-hot encoding
    one_hot_targets = []
    if "Race" in all_df.columns:
        all_df["Race"] = all_df["Race"].astype(str).fillna("missing")
        mapping["one_hot_columns"]["Race"] = sorted(all_df["Race"].unique().tolist())
        one_hot_targets.append("Race")

    all_df = pd.get_dummies(all_df, columns=one_hot_targets, prefix=one_hot_targets, dtype=np.int8)

    # Fill numeric missing values
    for col in all_df.columns:
        if col in [ID_COL, "_is_train"]:
            continue

        if all_df[col].dtype == "object":
            all_df[col] = all_df[col].astype(str).fillna("missing")
        else:
            med = all_df[col].median()
            if pd.isna(med):
                med = 0
            all_df[col] = all_df[col].fillna(med)

    # Last-resort: if any other object columns remain, drop them rather than silently using them.
    remain_obj = [c for c in all_df.columns if all_df[c].dtype == "object" and c != ID_COL]
    if remain_obj:
        mapping["dropped_unhandled_object_columns"] = remain_obj
        all_df = all_df.drop(columns=remain_obj)

    # Feature engineering after numeric conversion
    all_df = add_features(all_df)

    processed_train = all_df[all_df["_is_train"] == 1].copy()
    processed_test = all_df[all_df["_is_train"] == 0].copy()

    processed_train[TARGET] = y.values
    processed_test[TARGET] = np.nan

    processed_data = pd.concat([processed_train, processed_test], axis=0, ignore_index=True)

    first_cols = [ID_COL, "_is_train", TARGET]
    other_cols = [c for c in processed_data.columns if c not in first_cols]
    processed_data = processed_data[first_cols + other_cols]

    processed_train = processed_data[processed_data["_is_train"] == 1].drop(columns=["_is_train"]).copy()
    processed_test = processed_data[processed_data["_is_train"] == 0].drop(columns=["_is_train", TARGET]).copy()

    processed_data.to_csv(OUT_ALL, index=False)
    processed_train.to_csv(OUT_TRAIN, index=False)
    processed_test.to_csv(OUT_TEST, index=False)

    feature_cols = [c for c in processed_train.columns if c not in [ID_COL, TARGET]]
    Path(OUT_FEATURES).write_text("\n".join(feature_cols), encoding="utf-8")

    mapping["feature_count"] = len(feature_cols)
    mapping["processed_train_shape"] = list(processed_train.shape)
    mapping["processed_test_shape"] = list(processed_test.shape)

    with open(OUT_MAPPING, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print("Saved:", OUT_ALL)
    print("Saved:", OUT_TRAIN)
    print("Saved:", OUT_TEST)
    print("Saved:", OUT_MAPPING)
    print("Saved:", OUT_FEATURES)
    print("Processed train shape:", processed_train.shape)
    print("Processed test shape :", processed_test.shape)
    print("Feature count:", len(feature_cols))

    print()
    print("Dropped columns:", DROP_COLS)

    print()
    print("Compound mapping:")
    for k, v in COMPOUND_MAPPING.items():
        print(f"  {k:12s} -> {v}")

    if "Race" in mapping["one_hot_columns"]:
        print()
        print("Race one-hot columns:")
        for race in mapping["one_hot_columns"]["Race"]:
            print(f"  Race_{race}")


if __name__ == "__main__":
    main()
