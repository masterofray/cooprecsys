'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

import xgboost as xgb
import time
start_time = time.time()

params = {
    # --- Core Objectives ---
    'objective': 'rank:ndcg',
    'eval_metric': ['ndcg@5', 'map@5'],
    'learning_rate': 0.05,

    # --- Speed Optimization ---
    'tree_method': 'hist',        # Fast histogram-based splits
    'max_depth': 6,               # Shallower trees = much faster CPU training
    'gamma': 0.2,                       # Minimum loss reduction to make a split

    # --- Parallel Processing & Silence ---
    'nthread': -1,                # <--- Restored: Uses all available CPU cores
    'verbosity': 0,               # 0 is silent, 1 is warnings, 2 is info

    # --- Sampling & Regularization ---
    'subsample': 0.8,                   # Sample rows for each tree
    'colsample_bytree': 0.8,            # Sample features for each tree
    'colsample_bylevel': 0.7,           # Further sampling at each depth level

    # --- Regularization ---
    'lambda': 1.5,
    'alpha': 0.5,

    # --- Hardware & Reproducibility ---
    'tree_method': 'hist',
    'seed': 3
}

# Pro-tip: Ensure your data is sorted by group/user_id before this step!
dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(group_train)

dtest = xgb.DMatrix(X_test, label=y_test)
dtest.set_group(group_test)

# Increased rounds + Early Stopping
model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=400,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    #early_stopping_rounds=50,
    verbose_eval=50
)

eval_results = model_xgb.eval(dtest)
print(f"\nFinal Test Set Performance:\n{eval_results}")

end_time = time.time()
runtime = (end_time - start_time)/60
print(f"\n\nEstimated runtime: {runtime:.2f} minutes.")

model_save_path_ltr = 'xgboost_ltr_model.json'
model_xgb.save_model(model_save_path_ltr)
print(f"XGBoost LTR model saved to {model_save_path_ltr}")

# Perform 5-Fold Cross-Validation
Doing_the_KFold = False

if Doing_the_KFold:
    print("Starting Cross-Validation...")
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        nfold=5,
        stratified=False,
        shuffle=False,
        verbose_eval=50
    )

    best_round = cv_results['test-ndcg@5-mean'].idxmax()
    best_score = cv_results.loc[best_round, 'test-ndcg@5-mean']

    print(f"Best Round from CV: {best_round}")
    print(f"Best CV NDCG: {best_score}")
    
    