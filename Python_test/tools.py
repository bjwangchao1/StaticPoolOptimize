from inspect import getmembers
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pickle


def IV(x, y):
    crtab = pd.crosstab(x, y, margins=True)
    crtab.columns = ['good', 'bad', 'total']
    crtab['factor_per'] = crtab['total'] / len(y)
    crtab['bad_per'] = crtab['bad'] / crtab['total']
    crtab['p'] = crtab['bad'] / crtab.loc['All', 'bad']
    crtab['q'] = crtab['good'] / crtab.loc['All', 'good']
    crtab['woe'] = np.log(crtab['p'] / crtab['q'])
    crtab2 = crtab[abs(crtab.woe) != np.inf]

    crtab['IV'] = sum((crtab2['p'] - crtab2['q']) * np.log(crtab2['p'] / crtab2['q']))
    crtab.reset_index(inplace=True)
    crtab['varname'] = crtab.columns[0]
    crtab.rename(columns={crtab.columns[0]: 'var_level'}, inplace=True)
    crtab.var_level = crtab.var_level.apply(str)
    return crtab


def decision_bin(x, target):
    """
    决策树分箱
    :param x: columns of feature
    :param target: target feature
    :return: split dot
    """
    clf = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_leaf=0.05)
    x = np.array(x).reshape(len(x), 1)
    y = list(target)
    model = clf.fit(x, y)
    v_tree = getmembers(model.tree_)
    v_tree_thres = dict(v_tree)['threshold']
    v_tree_thres = sorted(list(v_tree_thres[v_tree_thres != -2]))
    split_p = [min(x)[0]] + v_tree_thres + [max(x)[0] + 1]
    return split_p


def com_ks(y_pred, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = np.max(tpr - fpr)
    return ks


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
    # save_figure(plt, title)
    plt.show()
    return plt


def load_model(path):
    with open(path, 'rb') as f:
        bst = pickle.load(f)
    return bst


def specification_fp(data):
    df = pd.DataFrame()
    df['英文变量名'] = data.columns
    for i in data.columns.tolist():
        if data[i].isnull().any():
            df.loc[df.英文变量名 == i, '是否允许缺失值'] = '是'
        else:
            df.loc[df.英文变量名 == i, '是否允许缺失值'] = '否'
        if data[i].nunique(dropna=False) >= 10:
            df.loc[df.英文变量名 == i, '值字典'] = str(data[i].unique()[:5].tolist())
        else:
            df.loc[df.英文变量名 == i, '值字典'] = str(data[i].unique().tolist())

    def field_astype(x):
        if x == 'int64':
            return 'int'
        elif x == 'float64':
            return 'numeric'
        else:
            return 'string'

    df['不同值个数'] = data.nunique(dropna=False).tolist()
    df['字段类型'] = data.dtypes.tolist()
    df['字段类型'] = df.字段类型.map(field_astype)
    return df
