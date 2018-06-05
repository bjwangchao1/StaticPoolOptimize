import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import mean_absolute_error, make_scorer

data = pd.read_csv(r'C:\Users\bjwangchao1\Desktop\建模大赛2\选手建模数据\process_data\f_model\f_model_feature.csv',
                   engine='python', encoding='utf-8')

x_train, x_test, y_train, y_test = train_test_split(data.drop(['ccx_id', 'target'], axis=1),
                                                    data.target,
                                                    test_size=0.3, random_state=42, stratify=data.target)

# 自定义评价函数
# def xg_eval_mae(yhat, dtrain):
#     y = dtrain.get_label()
#     return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))
#
#
# def mae_score(y_true, y_pred):
#     return mean_absolute_error(np.exp(y_true), np.exp(y_pred))
#
#
# mae_scorer = make_scorer(mae_score, greater_is_better=False)

"""
XGBoost参数调节
1.选择一组初始参数
2.改变max_depth和min_child_weight
3.调节gamma降低模型过拟合风险
4.调节subsample和colsample_bytree改变数据采样策略
"""

"""
xgboost参数
· 'booster':'gbtree',
· 'objective':'multi:softmax',多分类问题,
· 'num_class':10,类别数,与multisoftmax并用
· 'gamma':损失下降多少才进行分裂,
· 'max_depth':12,构建数的深度，越大越容易过拟合,
· 'lambda':2,控制模型复杂度的l2正则化参数，参数越大，越不容易过拟合,
· 'subsample':0.7,随机采样训练样本,
· 'colsample_bytree':0.7,生成树时的列采样,
· 'min_child_weight':3,孩子节点中最小样本权重和.如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
· 'silent':0,如果是1没有任何信息输出,
· 'eta':0.007,如同学习率,
· 'seed':1000,
· 'nthread':7,cpu线程数
"""


class XGBoostClassifier(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        if 'num_boost_round' in self.params:
            self.num_boost_round = self.params['num_boost_round']
            del self.params['num_boost_round']
        self.params.update(
            {'silent': 1, 'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 0,
             'nthread': -1})
        self.bst = None

    def fit(self, train_x, train_y, test_x, test_y):
        dtrain = xgb.DMatrix(train_x, label=train_y, missing=np.nan)
        dval = xgb.DMatrix(test_x, label=test_y)
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        self.bst = xgb.train(params=self.params, dtrain=dtrain, evals=watchlist, num_boost_round=self.num_boost_round,
                             early_stopping_rounds=50, verbose_eval=True)

    def predict(self, x_pred):
        d_pred = xgb.DMatrix(x_pred, missing=np.nan)
        return self.bst.predict(d_pred)

    def kfold(self, train_x, train_y, n_fold=5):
        dtrain = xgb.DMatrix(train_x, label=train_y)
        cv_rounds = xgb.cv(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                           nfold=n_fold, metrics='auc', maximize=False, early_stopping_rounds=50)
        return cv_rounds.iloc[-1, :]

    def plot_feature_importance(self):
        feat_imp = pd.Series(self.bst.get_fscore()).sort_values(ascending=False)
        feat_imp.plot(title='Feature Importance')
        plt.ylabel('Feature Importance Score')

    def get_params(self):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self


# bst = XGBoostClassifier(eta=0.05, colsample_bytree=0.75, subsample=0.75, max_depth=5, min_child_weight=30,
#                         num_boost_round=500, gamma=3)
#
# bst.fit(x_train, y_train, x_test, y_test)

# 树的深度与节点权重
bst = XGBoostClassifier(eta=0.05, colsample_bytree=0.75, subsample=0.75, num_boost_round=500, gamma=3)

xgb_parm_grid = {'max_depth': list(range(4, 9)), 'min_child_weight': list((1, 3, 6))}

grid = GridSearchCV(bst, param_grid=xgb_parm_grid, cv=5, scoring='roc_auc', n_jobs=-1)

grid.fit(x_train, y_train.values)
