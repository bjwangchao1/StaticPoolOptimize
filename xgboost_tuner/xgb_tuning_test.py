import pandas as pd
from xgboost_tuner.tuner import tune_xgb_params

if __name__ == '__main__':
    data = pd.read_csv(r'C:\Users\bjwangchao1\Desktop\finup_ecmm\mofei\电商模型更新样本\万象分更新\base_ecom_var.csv',
                       engine='python', encoding='utf-8')
    data_x = data.drop(['model_id', 'target'], axis=1)
    data_y = data['target']

    best_params, history = tune_xgb_params(
        cv_folds=2,
        label=data_y,
        metric_sklearn='roc_auc',
        metric_xgb='auc',
        n_jobs=4,
        objective='binary:logistic',
        random_state=2018,
        strategy='incremental',
        train=data_x,
        colsample_bytree_min=0.7,
        colsample_bytree_max=0.8,
        colsample_bytree_step=0.05,
        gamma_max=1,
        gamma_min=1,
        max_depth_min=1,
        max_depth_max=3,
        min_child_weight_max=6,
        min_child_weight_min=5,
        subsample_min=0.7,
        subsample_max=0.8,
        subsample_step=0.05
    )

    print(best_params, history)
