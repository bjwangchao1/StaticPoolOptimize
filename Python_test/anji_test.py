import pandas as pd
import numpy as np
import pickle
from Python_test.tools import IV, com_ks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = pd.read_csv(r'E:\安吉农商行\安吉农商行建模数据\anji_data.txt', engine='python', sep='\t')
target = pd.read_csv(r'E:\安吉农商行\df_smp.csv', engine='python')

df = pd.merge(data, target[['序号', 'if_bad_3']], on='序号', how='right')

trade = pd.read_csv(r'E:\安吉农商行\安吉农商行建模数据\trade_data.txt', engine='python', sep='\t')
trade['进账交易量总额'] = trade['20151231之前进账交易量总额'] + trade['20151231之后进账交易量总额']
trade['借记卡进账次数'] = trade['20151231之前借记卡进账次数'] + trade['20151231之后借记卡进账次数']
trade['借记卡出账次数'] = trade['20151231之前借记卡出账次数'] + trade['20151231之后借记卡出账次数']
trade['出账交易量总额'] = trade['20151231之前出账交易量总额'] + trade['20151231之后出账交易量总额']

df_test = pd.merge(df, trade, on='序号', how='left')

data = pd.read_csv(r'E:\安吉农商行\安吉农商行建模数据\anji_model_test.csv', engine='python')

col = ['period', 'guarantee_method', 'deposit_balance_meanf3', 'marriage', 'trade_amount_out',
       'education', 'identity', 'Private_customer_type', 'gender']

df = data[['id', 'if_bad_3'] + col]


# df2 = data[['id', 'if_bad_3', 'Private_customer_type', 'industry_orient']]
# df2['green'] = df2.industry_orient.map(lambda x: 1 if x in green else 0)


def f_edu_class(x):
    if x in ['初中', '中等专业技术学校', '技工学校', '小学', '文盲或半文盲']:
        return '中专及以下'
    elif x in ['高中']:
        return '高中'
    elif x in ['大专和专科学校']:
        return '专科'
    elif x in ['大学本科', '硕士']:
        return '本科及以上'
    else:
        return '其他'


def f_identity_class(x):
    if x in ['工人', '自由职业者', '企业管理人员', '农民']:
        return '其他'
    else:
        return x


def f_borrower_type_class(x):
    if x in ['农户', '农村个体工商户（个人）']:
        return '农户'
    else:
        return x


def f_marriage_class(x):
    if x in ['已婚', '未说明婚姻状况']:
        return '已婚'
    elif x in ['离婚', '丧偶']:
        return '离婚或丧偶'
    else:
        return x


df.education = df.education.apply(f_edu_class)
df.identity = df.identity.apply(f_identity_class)
# df.borrower_type = df.borrower_type.apply(f_borrower_type_class)
df.marriage = df.marriage.apply(f_marriage_class)

# 计算IV
ls = pd.DataFrame()
for i in [x for x in df.columns if x not in ['id', 'if_bad_3']]:
    if i not in ['deposit_balance_meanf3', 'trade_amount_out', 'debit_in_times']:
        moment = IV(df[i].fillna(-99), df['if_bad_3'])
        ls = ls.append(moment)
    else:
        pass

dep = IV(pd.cut(df.deposit_balance_meanf3, bins=[0, 25, 600, 1400, 57000], right=False), df.if_bad_3)
# dep = IV(pd.cut(df.debit_in_times, bins=[0, 4, 70, 1300], right=False), df.if_bad_3)
dep = IV(pd.cut(df.trade_amount_out, bins=[0, 8.5, 150, 5505], right=False), df.if_bad_3)

ls_dict = ls[['var_level', 'woe']]
ls_dict = ls_dict.set_index('var_level')
ls_dict.drop('All', axis=0, inplace=True)
ls_dict = ls_dict.to_dict().get('woe')

df.deposit_balance_meanf3 = pd.cut(df.deposit_balance_meanf3, bins=[0, 25, 600, 1400, 57000], right=False).astype(str)
df.trade_amount_out = pd.cut(df.trade_amount_out, bins=[0, 8.5, 150, 5505], right=False).astype(str)


def f_trans_dict(x):
    if x in ls_dict.keys():
        x = ls_dict.get(x)
    return x


df1 = df.copy()

df1 = df1.applymap(f_trans_dict)
df1 = df1.replace(-np.inf, 0)
df1.loc[df1.gender.isnull(), 'gender'] = -0.3184347319882966

# 建立模型
x_train, x_test, y_train, y_test = train_test_split(
    df1.drop(['if_bad_3', 'period'], axis=1), df1.if_bad_3,
    test_size=0.2, random_state=0)
clf = LogisticRegression(penalty='l1', C=1.0)

clf.fit(x_train.drop('id', axis=1), y_train)

# 模型效果
print('train_AUC:', roc_auc_score(y_train, clf.predict_proba(x_train.drop('id', axis=1))[:, 1]))
print('test_AUC:', roc_auc_score(y_test, clf.predict_proba(x_test.drop('id', axis=1))[:, 1]))

print('xgb-train-ks', com_ks(clf.predict_proba(x_train.drop('id', axis=1))[:, 1], y_train))
print('xgb-test-ks', com_ks(clf.predict_proba(x_test.drop('id', axis=1))[:, 1], y_test))

train_score = pd.DataFrame(
    {'id': x_train.id, 'target': y_train, 'prob': clf.predict_proba(x_train.drop('id', axis=1))[:, 1]})
test_score = pd.DataFrame(
    {'id': x_test.id, 'target': y_test, 'prob': clf.predict_proba(x_test.drop('id', axis=1))[:, 1]})

anji_score = pd.concat([train_score, test_score], axis=0, ignore_index=True)
anji_score['score'] = anji_score.prob.map(lambda x: score(x))


def score(prob):
    sco = 40 - (7 / np.log(2)) * np.log(prob / (1 - prob))
    return round(sco, 1)


# 保存模型
with open('anji_model.txt', 'wb') as f:
    pickle.dump(clf, f)

green = ['其他农业',
         '茶及其他饮料作物种植',
         '金属废料和碎屑加工处理',
         '非金属废料和碎屑加工处理',
         '农村基础设施建设',
         '个人住房改造贷款',
         '其他林业服务',
         '城乡市容管理',
         '林木育苗',
         '游乐园',
         '内河货物运输',
         '水污染治理',
         '市政设施管理',
         '农业技术推广服务',
         '其他水利管理业',
         '市政道路工程建筑',
         '林木育种',
         '森林经营和管护',
         '农田基本设施建设',
         '其他园艺作物种植',
         '造林和更新',
         '游览景区管理',
         '新材料技术推广服务',
         '环境保护专用设备制造',
         '城市轨道交通设备制造',
         '绿化管理',
         '光伏设备及元器件制造',
         '其他农、林、牧、渔业机械制造',
         '水资源管理',
         '葡萄种植',
         '其他水果种植',
         '太阳能发电',
         '河湖治理及防洪设施工程建筑',
         '花卉种植',
         '风能原动设备制造',
         '公路旅客运输',
         '再生物资回收与批发',
         '坚果种植',
         '燃气、太阳能及类似能源家用器具制造',
         '其他城市公共交通运输',
         '环境卫生管理',
         '精制茶加工',
         '节能技术推广服务',
         '仁果类和核果类水果种植',
         '农业科学研究和试验发展',
         '其他清洁服务',
         '农业机械服务',
         '农林牧渔专用仪器仪表制造',
         '沿海货物运输',
         '固体废物治理',
         '电力供应',
         '含油果种植',
         '其他水上运输辅助活动',
         '其他自然保护',
         '其他电力生产',
         '农业机械批发',
         '港口及航运设施工程建筑',
         '燃气生产和供应业',
         '水源及供水设施工程建筑',
         '食用菌种植',
         '水力发电',
         '热力生产和供应',
         '货运港口',
         '铁路旅客运输',
         '船用配套设备制造',
         '水资源专用机械制造',
         '种子批发',
         '管道工程建筑',
         '铁路工程建筑',
         '水路航道建设和管理',
         '公共电汽车客运']  # 绿色属性


def load_model(path):
    with open(path, 'rb') as f:
        bst = pickle.load(f)
    return bst


clf = load_model(r'E:\安吉农商行\安吉农商行建模数据\anji_model.txt')

data1 = pd.read_excel(r'E:\安吉农商行\新建 Microsoft Excel 工作表.xlsx', sheet_name='Sheet1')

"""
count       419.000000
mean       2513.100239
std       13583.420818
min           1.000000
25%           9.000000
50%          75.000000
75%        3000.000000
max      200256.000000
"""




plot_ks_line(y_test, clf.predict_proba(x_test.drop('id', axis=1))[:, 1], title='ks-curve(test)')
plot_ks_line(y_train, clf.predict_proba(x_train.drop('id', axis=1))[:, 1], title='ks-curve(train)')



plot_roc_line(y_train, clf.predict_proba(x_train.drop('id', axis=1))[:, 1], title='AUC-curve(train)')

plot_roc_line(y_test, clf.predict_proba(x_test.drop('id', axis=1))[:, 1], title='AUC-curve(test)')


anji_score=pd.read_csv(r'E:\安吉农商行\安吉农商行建模数据\anji_score_1226.csv',engine='python')



