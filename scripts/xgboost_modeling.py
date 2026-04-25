from xgboost import XGBClassifier 
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pandas as pd

# Path Setup
data_dir = "./data/feature_extracted"
pilots = [5,6,9,10,11,12,13,14,17,18,19,21,22,23,24,25,26]
runs = ["CA", "DA", "SS","LOFT"]

all_dfs = []
for p in pilots:
    for r in runs:
        file_path = f"{data_dir}/{p}/{p}_{r}_features.csv"

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            df['subject_id'] = p
            df['file_type'] = r

            all_dfs.append(df)
            print(f'appended: pilot {p} and run {r}')

master_df = pd.concat(all_dfs, ignore_index=True)
del all_dfs
print("-------Master df created-------")

# Data
# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y = np.array([1, 2, 1, 2])
# groups = np.array([1, 1, 2, 2])
X = master_df.drop(columns=['subject_id', 'file_type', 'window_id', 'label'])
y = master_df['label']
groups = master_df['subject_id']
print("-------Data Split-------")

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("-------Label encoded output-------")

# Create XGBoost classifier model instance
num_unique_states = len(np.unique(y_encoded))
bst = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=1, objective='multi:softmax', num_class=num_unique_states)

# Setup Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# Perform Cross-Validation 
print("-------Performing model fitting + cross validation-------")
scores = cross_val_score(bst, X, y_encoded, cv=logo, scoring='accuracy', groups=groups)

print(f"Average Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation: {scores.std():.4f}")
print(scores)
