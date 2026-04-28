from xgboost import XGBClassifier 
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
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
            print(f'appended: pilot {p} and run {r}', flush=True)

master_df = pd.concat(all_dfs, ignore_index=True)
del all_dfs
print("-------Master df created-------", flush=True)

# Data
X = master_df.drop(columns=['subject_id', 'file_type', 'window_id', 'label'])
y = master_df['label']
groups = master_df['subject_id']

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_unique_states = len(np.unique(y_encoded))
param_grid = {
    'n_estimators': [200, 250, 300],
    'max_depth': [2, 3],
    'learning_rate': [0.05, 0.07, 0.1],
    'subsample': [0.6, 0.7],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'min_child_weight': [8, 10, 15],
    'gamma': [0, 0.1],
}

total_combos = np.prod([len(v) for v in param_grid.values()])
print(f"Total combinations: {total_combos}", flush=True)
print(f"Total fits: {total_combos * len(np.unique(groups))}", flush=True)

# Base model
bst = XGBClassifier(
    objective='multi:softmax',
    num_class=num_unique_states,
    eval_metric='mlogloss'
)

# Setup LOGO-CV
logo = LeaveOneGroupOut()

# Grid Search
print("-------Starting grid search-------", flush=True)
search = GridSearchCV(
    bst, param_grid, cv=logo,
    scoring='f1_weighted', n_jobs=-1, verbose=3,
    return_train_score=True
)
search.fit(X, y_encoded, groups=groups)

print('--- Grid Search Completed! ---', flush=True)

print(f"\nBest Params: {search.best_params_}", flush=True)
print(f"Best CV Weighted F1: {search.best_score_:.4f}", flush=True)
print("All done!", flush=True)

best_model = search.best_estimator_
os.makedirs("./model", exist_ok=True)
joblib.dump(best_model, "./model/best_xgb_model.pkl")
print("Best model saved!", flush=True)