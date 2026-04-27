from xgboost import XGBClassifier 
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, RandomizedSearchCV
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
X = master_df.drop(columns=['subject_id', 'file_type', 'window_id', 'label'])
y = master_df['label']
groups = master_df['subject_id']

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# set hyperparams for randomsearch CV
num_unique_states = len(np.unique(y_encoded))
param_grid = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.03, 0.05, 0.07],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [8, 10, 15],
}

# Create XGBoost classifier model instance
bst = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=1, objective='multi:softmax', num_class=num_unique_states)

# Setup Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# perform optimal hyperparameter search
search = RandomizedSearchCV(
    bst, param_grid, n_iter=50, cv=logo,
    scoring='accuracy', n_jobs=-1, verbose=1, 
    random_state=42, return_train_score=True  
)
search.fit(X, y_encoded, groups=groups)

results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values('mean_test_score', ascending=False)

# Keep only useful columns
cols = ['mean_test_score', 'std_test_score', 'mean_train_score'] + \
       [c for c in results_df.columns if c.startswith('param_')]
results_df[cols].to_csv("./katy/hyperparam_search_results.csv", index=False)
print('--- Random Hyperparam Search Completed! ---')

print(f"\nBest Params: {search.best_params_}")
print(f"Best CV Accuracy: {search.best_score_:.4f}")
print("All done!")