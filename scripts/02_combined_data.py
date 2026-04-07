#!/usr/bin/env python3

import os
import pandas as pd

RAW_FOLDER = "data/raw"
CLEAN_FOLDER = "data/clean"

#1 Load and combine csv files

controlled_states_data = []
loft_data = []

CONTROLLED_STATES_OUTPUT = os.path.join(CLEAN_FOLDER, "combined_controlled_states.csv")
LOFT_OUTPUT = os.path.join(CLEAN_FOLDER, "combined_loft.csv")

os.makedirs(CLEAN_FOLDER, exist_ok=True)

for root, dirs, files in os.walk(RAW_FOLDER):
	for file in files:
		if file.endswith(".csv"):
			path = os.path.join(root, file)
			print(f"Loading {path}")

			df = pd.read_csv(path)
			df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

			if "_loft" in file.lower():
				df["dataset"] = "loft"
				df["state"] = "unknown"
				loft_data.append(df)

			else:
				df["dataset"] = "controlledstates"

				if "CA" in file:
					df["state"] = "CA"
				elif "DA" in file:
					df["state"] = "DA"
				elif "SS" in file:
					df["state"] = "SS"
				else:
					df["state"] = "unknown"

				controlled_states_data.append(df)

			df["source_file"] = file

#2 Combine

controlled_states_df = pd.concat(controlled_states_data, ignore_index=True)
loft_df = pd.concat(loft_data, ignore_index=True)

#3 Clean

controlled_states_df = controlled_states_df.ffill().bfill()
loft_df = loft_df.ffill().bfill()

#4 Save cleaned data

controlled_states_df.to_csv(CONTROLLED_STATES_OUTPUT, index=False)
loft_df.to_csv(LOFT_OUTPUT, index=False)

print("Done!")
print("Controlled states rows:", len(controlled_states_df))
print("Loft rows:", len(loft_df))
