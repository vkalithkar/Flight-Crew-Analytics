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

# --- Save model stats ---
cv_results = pd.DataFrame(search.cv_results_)
n_folds = len(np.unique(groups))  # LOGO = one fold per subject

best_idx = search.best_index_
fold_test_cols  = [f"split{i}_test_score"  for i in range(n_folds)]
fold_train_cols = [f"split{i}_train_score" for i in range(n_folds)]

best_fold_test_scores  = cv_results.loc[best_idx, fold_test_cols].values.astype(float)
best_fold_train_scores = cv_results.loc[best_idx, fold_train_cols].values.astype(float)

lines = []
lines.append("=" * 60)
lines.append("MODEL STATS — XGBoost LOGO-CV Grid Search")
lines.append("=" * 60)

lines.append("\n--- Hyperparameter Search Space ---")
for k, v in param_grid.items():
    lines.append(f"  {k}: {v}")
lines.append(f"\n  Total combinations : {total_combos}")
lines.append(f"  Total fits         : {total_combos * n_folds}")

lines.append("\n--- Best Hyperparameters ---")
for k, v in search.best_params_.items():
    lines.append(f"  {k}: {v}")

lines.append("\n--- Overall CV Performance (Best Params) ---")
lines.append(f"  Mean test  weighted-F1 : {search.best_score_:.4f}")
lines.append(f"  Std  test  weighted-F1 : {best_fold_test_scores.std():.4f}")
lines.append(f"  Mean train weighted-F1 : {best_fold_train_scores.mean():.4f}")
lines.append(f"  Std  train weighted-F1 : {best_fold_train_scores.std():.4f}")

lines.append(f"\n--- Per-Fold Scores (Best Params, {n_folds} folds = leave-one-pilot-out) ---")
lines.append(f"  {'Fold (pilot)':<15} {'Train F1':>10} {'Test F1':>10}")
lines.append(f"  {'-'*37}")
for i, pilot in enumerate(np.unique(groups)):
    train_s = best_fold_train_scores[i]
    test_s  = best_fold_test_scores[i]
    lines.append(f"  Pilot {str(pilot):<9} {train_s:>10.4f} {test_s:>10.4f}")

lines.append("\n--- Top 10 Configurations by Mean Test F1 ---")
top10 = (
    cv_results
    .sort_values("mean_test_score", ascending=False)
    .head(10)
    [["mean_test_score", "std_test_score", "mean_train_score", "params"]]
    .reset_index(drop=True)
)
for i, row in top10.iterrows():
    lines.append(f"\n  Rank {i+1}:")
    lines.append(f"    mean_test_F1  = {row['mean_test_score']:.4f}  (±{row['std_test_score']:.4f})")
    lines.append(f"    mean_train_F1 = {row['mean_train_score']:.4f}")
    for k, v in row["params"].items():
        lines.append(f"    {k}: {v}")

lines.append("\n" + "=" * 60)

stats_path = "./model/model_stats.txt"
with open(stats_path, "w") as f:
    f.write("\n".join(lines))
print(f"Model stats saved to {stats_path}", flush=True)