#!/usr/bin/env python3

import os
import pandas as pd

RAW_FOLDER = "data/raw"
CLEAN_FOLDER = "data/clean"

#1 Load and combine csv files

train_data = []
loft_data = []

TRAIN_OUTPUT = os.path.join(CLEAN_FOLDER, "cleaned_train.csv")
LOFT_OUTPUT = os.path.join(CLEAN_FOLDER, "cleaned_loft.csv")

os.makedirs(CLEAN_FOLDER, exist_ok=True)

for root, dirs, files in os.walk(RAW_FOLDER):
    for file in files:
        if file.endswith(".csv"):
            path = os.path.join(root, file)
            print(f"Loading {path}")

            df = pd.read_csv(path)

        if "_loft" in file.lower():
            df["dataset"] = "loft"
            df["state"] = "unknown"
            loft_data.append(df)

        else:
            df["dataset"] = "train"

            if "CA" in file:
                df["state"] = "CA"
            elif "DA" in file:
                df["state"] = "DA"
            elif "SS" in file:
                df["state"] = "SS"

            train_data.append(df)

        df["source_file"] = file

#2 Combine

train_df = pd.concat(train_data, ignore_index=True)
loft_df = pd.concat(loft_data, ignore_index=True)

#3 Clean

train_df = train_df.ffill().bfill()
loft_df = loft_df.ffill().bfill()

#4 Save cleaned data

train_df.to_csv(TRAIN_OUTPUT, index=False)
loft_df.to_csv(LOFT_OUTPUT, index=False)

print("Done!")
print("Train rows:", len(train_df))
print("Loft rows:", len(loft_df))
