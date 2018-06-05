import pandas as pd
import numpy as np
# import os, random
# import pickle
from Python_test.tools import com_ks
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import xgboost as xgb

data = pd.read_csv(r'C:\Users\bjwangchao1\Desktop\车300\f_data_target1.csv', engine='python')
car_data = pd.read_csv(r'C:\Users\bjwangchao1\Desktop\车300\che300_model_data.csv', engine='python', encoding='utf-8')
sex_dict = {'男': 1, '女': 0, np.nan: 1}
car_data.sex = car_data.sex.map(sex_dict)
car_data_dum = pd.get_dummies(car_data, columns=['marry_status', 'degree'], dummy_na=True)

all_data = pd.merge(car_data_dum, data.drop('ever_M3p', axis=1), on='PROJECT_NO', how='inner')
all_data1 = pd.merge(car_data, data.drop('ever_M3p', axis=1), on='PROJECT_NO', how='inner')

x_train, x_test, y_train, y_test = train_test_split(data.drop(['PROJECT_NO', 'ever_M3p'], axis=1), data.ever_M3p,
                                                    test_size=0.2, random_state=1024)

x_train1, x_test1, y_train1, y_test1 = train_test_split(car_data_dum.drop(['PROJECT_NO', 'ever_M3p'], axis=1),
                                                        car_data_dum.ever_M3p,
                                                        test_size=0.2, random_state=1024)
x_train2, x_test2, y_train2, y_test2 = train_test_split(all_data.drop(['PROJECT_NO', 'ever_M3p'], axis=1),
                                                        all_data.ever_M3p,
                                                        test_size=0.2, random_state=1024)

dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_test, label=y_test)

dtrain1 = xgb.DMatrix(x_train1, label=y_train1)
dval1 = xgb.DMatrix(x_test1, label=y_test1)

dtrain2 = xgb.DMatrix(x_train2, label=y_train2)
dval2 = xgb.DMatrix(x_test2, label=y_test2)

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 2,
    'lambda': 10,
    'gamma': 0,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 5,
    'eta': 0.025,
    'seed': 0,
    'nthread': 20,
    'silent': 1
}

params1 = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 2,
    'lambda': 10,
    'gamma': 0,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 5,
    'eta': 0.025,
    'seed': 0,
    'nthread': 20,
    'silent': 1
}
params2 = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 2,
    'lambda': 220,
    'gamma': 0,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 5,
    'eta': 0.025,
    'seed': 0,
    'nthread': 20,
    'silent': 1
}

watchlist = [(dtrain, 'train'), (dval, 'val')]

watchlist1 = [(dtrain1, 'train'), (dval1, 'val')]

watchlist2 = [(dtrain2, 'train'), (dval2, 'val')]

bst = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist, early_stopping_rounds=50, verbose_eval=True)

bst1 = xgb.train(params1, dtrain1, num_boost_round=500, evals=watchlist1, early_stopping_rounds=50, verbose_eval=True)

bst2 = xgb.train(params2, dtrain2, num_boost_round=1500, evals=watchlist2, early_stopping_rounds=50, verbose_eval=True)

print('未加车300变量测试集:', com_ks(bst2.predict(dval2), y_test2))
print('未加车300变量训练集', com_ks(bst2.predict(dtrain2), y_train2))

'''
未加车300变量测试集: 0.454650770916
未加车300变量训练集 0.517475229888
'''
plot_roc_line(y_test2, bst2.predict(dval2), title='auc-curve(test)')

plot_ks_line(y_test2, bst2.predict(dval2), title='ks-curve(test)')
plot_ks_line(y_train2, bst2.predict(dtrain2), title='ks-curve(train)')


def plot_ks_line(y_true, y_pred, title='ks-curve', detail=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.plot(tpr, label='tpr-line')
    plt.plot(fpr, label='fpr-line')
    plt.plot(tpr - fpr, label='KS-line')
    # 设置x的坐标轴为0-1范围
    plt.xticks(np.arange(0, len(tpr), len(tpr) // 10), np.arange(0, 1.1, 0.1))

    # 添加标注
    x0 = np.argmax(tpr - fpr)
    y0 = np.max(tpr - fpr)
    plt.scatter(x0, y0, color='black')  # 显示一个点
    z0 = thresholds[x0]  # ks值对应的阈值
    plt.text(x0 - 2, y0 - 0.12, ('(ks: %.4f,\n th: %.4f)' % (y0, z0)))

    if detail:
        # plt.plot([x0,x0],[0,y0],'b--',label=('thresholds=%.4f'%z0)) #在点到x轴画出垂直线
        # plt.plot([0,x0],[y0,y0],'r--',label=('ks=%.4f'%y0)) #在点到y轴画出垂直线
        plt.plot(thresholds[1:], label='thresholds')
        t0 = thresholds[np.argmin(np.abs(thresholds - 0.5))]
        t1 = list(thresholds).index(t0)
        plt.scatter(t1, t0, color='black')
        plt.plot([t1, t1], [0, t0])
        plt.text(t1 + 2, t0, 'thresholds≈0.5')

        tpr0 = tpr[t1]
        plt.scatter(t1, tpr0, color='black')
        plt.text(t1 + 2, tpr0, ('tpr=%.4f' % tpr0))

        fpr0 = fpr[t1]
        plt.scatter(t1, fpr0, color='black')
        plt.text(t1 + 2, fpr0, ('fpr=%.4f' % fpr0))

    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(plt, title)
    plt.show()
    return plt


def plot_roc_line(y_true, y_pred, title='ROC-curve'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    ks = np.max(tpr - fpr)
    plt.plot(fpr, tpr)  # ,label=('auc= %.4f'%roc_auc)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.text(0.7, 0.45, ('auc= %.4f \nks  = %.4f' % (roc_auc, ks)))

    plt.title(title)
    save_figure(plt, title)
    plt.show()
    return plt
