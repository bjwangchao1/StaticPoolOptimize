import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import pickle
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore')

data = pd.read_csv(r'C:\Users\bjwangchao1\Desktop\建模大赛2\选手建模数据\process_data\ecom_data.csv', engine='python')

data.drop(['mobile_hourbin0509_cntorder_desc_distcount', 'mobile_hourbin0912_cntorder_desc_distcount',
           'mobile_hourbin1214_cntorder_desc_distcount', 'mobile_hourbin1418_cntorder_desc_distcount',
           'mobile_hourbin1822_cntorder_desc_distcount', 'mobile_hourbin2224_cntorder_desc_distcount',
           'mobile_hourbin0005_cntorder_desc_distcount', 'mobile_hourbin00_cntorder_desc_distcount'], axis=1,
          inplace=True)

x_train, x_test, y_train, y_test = train_test_split(data.drop(['ccx_id', 'target'], axis=1),
                                                    data.target,
                                                    test_size=0.3, random_state=0, stratify=data.target)

x_train = x_train.loc[:, bst.get_fscore().keys()]
x_test = x_test.loc[:, bst.get_fscore().keys()]

dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_test, label=y_test)

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 1,
    'lambda': 300,
    'gamma': 2,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 30,
    'eta': 0.05,
    'seed': 0,
    'nthread': -1,
    'silent': 1
}

watchlist = [(dtrain, 'train'), (dval, 'val')]
# xgb.cv(params, dtrain, num_boost_round=1500, nfold=5, metrics='auc', early_stopping_rounds=50)  # cv�ӽ�0.79
bst = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist, early_stopping_rounds=50, verbose_eval=True)

len(bst.get_fscore())
len(bst.feature_names)

# 保存模型
with open('model_ecom.txt', 'wb') as f:
    pickle.dump(bst, f)


def model_data(test):
    dtest = xgb.DMatrix(test, missing=np.nan)
    return dtest


ecom_prob_a = pd.DataFrame(
    {'ccx_id': data.ccx_id, 'target': data.target, 'prob': bst.predict(model_data(data[bst.feature_names]))})

ecom_prob_b = pd.DataFrame(
    {'ccx_id': b_ecom_data.id, 'prob': bst.predict(model_data(b_ecom_data[bst.feature_names]))})
"""
prob       float64
is_sh      float64
is_mngt    float64
is_gt      float64
var20      float64
var21      float64
var22      float64
var23      float64
var24      float64
var25      float64               
var26      float64
var27      float64
var28      float64
var29      float64
var30      float64
var31      float64
var32      float64
var33      float64
var34      float64
var35      float64
var36      float64
var37      float64
var38      float64
var39      float64
var40      float64
var41      float64
var42      float64
var43      float64
var44      float64
var45      float64
            ...   
var2241    float64
var2242    float64
var2243    float64
var2244    float64
var2245    float64
var2246    float64
var2247    float64
var2248    float64
var2249    float64
var2250    float64
var2251    float64
var2252    float64
var2253    float64
var2254    float64
var2255    float64
var2256    float64
var2257    float64
var2258    float64
var2259    float64
var2260    float64
var2261    float64
var2262    float64
var2263    float64
var2264    float64
var2265    float64
var2266    float64
var2267    float64
var2268    float64
var2269    float64
var2270    float64
"""

