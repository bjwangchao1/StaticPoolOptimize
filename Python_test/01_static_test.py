# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:41:44 2017

@author: lenovo
"""
import pymysql
import json
import flask
from flask import request
import pandas as pd
import numpy as np
from scipy.stats import lognorm
from scipy.stats import beta
from sklearn.metrics import mean_squared_error
import random
import time

server = flask.Flask(__name__)


@server.route('/staticPoolAnalysisApi', methods=['post'])
def f_static_pool_analysis():
    try:
        ### 读入数据
        input_pid = request.data.decode()
        input_pid = '604A3C6F78YLDXA-3'
        db_conn = pymysql.connect(host='192.168.100.35', port=3306, user='u_ccx_axis', password='WNyG^!^MwGp9IXyB',
                                  db='ccx_axis')
        db_cursor = db_conn.cursor()
        db_sql = "SELECT a.pass_time, a.iid AS lend_request_id, a.signed_amount, a.period, a.phase, a.due_date, a.repaid_time, a.due_principal AS schMnthPrincipal, a.repaid_principal AS repay_principal \
        FROM package_asset_his b LEFT JOIN repayplan_his a ON b.poid = a.poid AND b.iid = a.iid WHERE b.pid = "
        db_cursor.execute(db_sql + '\'' + input_pid + '\'')
        df_repay = pd.DataFrame(list(db_cursor.fetchall()),
                                columns=['pass_time', 'lend_request_id', 'signed_amount', 'period', 'phase', 'due_date',
                                         'repaid_time', 'schMnthPrincipal', 'repay_principal']).fillna(np.nan)
        db_conn.close()
        #        df_repay = pd.read_csv('C:/Users/lenovo/Desktop/静态池分析/04_数据样例_还款分布表.csv', engine='python')
        #        df_repay = df_repay.rename(columns={'进件号': 'lend_request_id'})
        #        df_repay = df_repay[['pass_time','lend_request_id', 'signed_amount', 'period','phase', 'due_date', 'repaid_time', 'schMnthPrincipal','repay_principal']]

        ### 预处理
        ## to datetime
        df_repay['pass_time'] = pd.to_datetime(df_repay['pass_time'])
        df_repay['due_date'] = pd.to_datetime(df_repay['due_date'])
        df_repay['repaid_time'] = pd.to_datetime(df_repay['repaid_time'])
        ## 月份
        df_repay['放款月份'] = df_repay['pass_time'].map(lambda x: x.strftime("%Y-%m"))
        df_repay['报告月份'] = df_repay['due_date'].map(lambda x: x.strftime("%Y-%m"))
        df_repay['实还月份'] = df_repay['repaid_time'].map(lambda x: np.nan if pd.isnull(x) else x.strftime("%Y-%m"))
        latest_repaid_time = df_repay.repaid_time.max()
        ## 截止时间限制
        df_repay = df_repay.loc[df_repay.due_date < pd.to_datetime(format(latest_repaid_time, '%Y-%m')), :]
        ## 计算逐期的逾期状态、剩余本金
        df_repay = f_calculate_status_etc_2(df_repay, 'lend_request_id', 'phase', 'due_date', 'repaid_time',
                                            'signed_amount', 'schMnthPrincipal', 'repay_principal')
        ##
        #        df_test = df_repay

        ### 静态池分析
        f1 = lambda x: str(x.year - 1) + '-' + '12' if x.month == 1 else str(x.year) + '-' + str(x.month - 1)
        end_month = f1(latest_repaid_time)
        df_pd_stats, df_pd_density_curves, df_pd_timing_curves = f_pd_analysis(df_repay, end_month=end_month)
        df_pr_stats, df_pr_density_curves, df_pr_timing_curves = f_pr_analysis(df_repay, end_month=end_month)
        df_pp_stats, df_pp_density_curves, df_pp_timing_curves = f_pp_analysis(df_repay, end_month=end_month)

        ### 输出
        ret_all = {
            "pdDist": df_pd_density_curves.to_dict(orient='records'),
            "pdTimeDist": df_pd_timing_curves.to_dict(orient='records'),
            "pdParam": df_pd_stats.to_dict(orient='records')[0],
            "prDist": df_pr_density_curves.to_dict(orient='records'),
            "prTimeDist": df_pr_timing_curves.to_dict(orient='records'),
            "prParam": df_pr_stats.to_dict(orient='records')[0],
            "ppDist": df_pp_density_curves.to_dict(orient='records'),
            "ppTimeDist": df_pp_timing_curves.to_dict(orient='records'),
            "ppParam": df_pp_stats.to_dict(orient='records')[0]
        }
        return json.dumps(ret_all)

    except Exception as e:
        ret = {"code": 502, "msg": "计算失败", "error_msg": str(e)}
        return json.dumps(ret, ensure_ascii=False)


# %%
## 计算月末状态、剩余本金等
def f_calculate_status_etc_2(df_repay, col_contract_no, col_term, col_due_date, col_repay_date, col_total_principal,
                             col_due_principal, col_repay_principal):
    # 计算每期的月末逾期状态、月末剩余本金，以及首次违约的周期（2017-11-25）
    # 输入：
    # “长格式”的还款记录数据框df_repay，
    # 包含的列有合同号（col_contract_no）、周期（col_term）、
    # 应还日期（col_due_date）、实还日期（col_repay_date）、总本金（col_total_principal）、实还本金（col_repay_principal）
    # 输出：
    # 原数据框增加了月末状态（status_monthend）、月末剩余本金（principal_balance）、是否首次违约（if_first_default）

    ### to datetime and add column monthend
    df_repay[col_due_date] = pd.to_datetime(df_repay[col_due_date])
    df_repay[col_repay_date] = pd.to_datetime(df_repay[col_repay_date])
    df_repay['monthend'] = df_repay[col_due_date].apply(
        lambda x: pd.to_datetime(str(x.year + 1) + '-' + '01') if x.month == 12 else pd.to_datetime(
            str(x.year) + '-' + str(x.month + 1)))

    ### calculate the relevant
    current_no = ''
    arr_if_first_default = np.zeros(len(df_repay))
    arr_status_monthend = np.zeros(len(df_repay))
    arr_principal_balance = np.zeros(len(df_repay))
    for i in range(len(df_repay)):
        # 当期周期
        current_term = df_repay[col_term].iloc[i]
        # 历期是否有还款
        sr_if_repaid = df_repay['monthend'].iloc[i] > df_repay[col_repay_date].iloc[(i - current_term + 1):(i + 1)]
        # 剩余本金
        arr_principal_balance[i] = df_repay[col_total_principal].iloc[i] - sum(
            df_repay[col_repay_principal].iloc[(i - current_term + 1):(i + 1)] * sr_if_repaid)
        # 历期是否全额还款（部分早偿待处理）
        sr_if_repaid_full = sr_if_repaid & np.logical_or((df_repay[col_repay_principal].iloc[
                                                          (i - current_term + 1):(i + 1)] >= df_repay[
                                                                                                 col_due_principal].iloc[
                                                                                             (i - current_term + 1):(
                                                                                                 i + 1)]),
                                                         (arr_principal_balance[(i - current_term + 1):(i + 1)] <= 1))
        # 月末状态
        arr_status_monthend[i] = current_term - sum(sr_if_repaid_full)
        # 是否首次违约
        if (arr_status_monthend[i] == 2) & (df_repay[col_contract_no].iloc[i] != current_no):
            arr_if_first_default[i] = 1
            current_no = df_repay[col_contract_no].iloc[i]

    df_repay['status_monthend'] = arr_status_monthend
    df_repay['principal_balance'] = arr_principal_balance
    df_repay['if_first_default'] = arr_if_first_default

    ### return
    return (df_repay)


## 静态池处理方法
def f_delta_loss_curve_1(df_static_pool, col_pool_month, col_term, col_pool_total, col_delta_default, end_month):
    # 静态池计算方法（delta loss curve method） (2017-11-23)
    # 输入：
    # 长格式的静态池数据框（df_static_pool），
    # 包含列放款月（col_pool_month）、周期（col_term）、月放款总额（col_pool_total）、当期违约金额（col_delta_default）
    # 输出：
    # 到期违约率，以及中间结果

    ### calculate the delta default rate
    def myfun1(sr, end_month=end_month):
        f1 = lambda x: pd.to_datetime(x).year * 12 + pd.to_datetime(x).month
        dif_month = f1(end_month) - f1(sr.name)
        sr[:(dif_month + 1)] = sr[:(dif_month + 1)].fillna(0)
        return sr

    df_delta_default_amount = df_static_pool.pivot(col_term, col_pool_month, col_delta_default).apply(myfun1)
    sr_pool_total_amount = df_static_pool.groupby(col_pool_month)[col_pool_total].first()
    df_delta_default_rate = df_delta_default_amount / sr_pool_total_amount

    ### calculate the expected lifetime default rate
    sr_avg_delta_default_rate = df_delta_default_rate.mean(axis=1)
    sr_cum_delta_default_rate = sr_avg_delta_default_rate.fillna(0).cumsum()
    sr_delta_timing_curve = sr_avg_delta_default_rate / sr_cum_delta_default_rate.values[-1:]
    sr_cum_timing_curve = sr_cum_delta_default_rate / sr_cum_delta_default_rate.values[-1:]
    df_cum_default_rate = df_delta_default_rate.cumsum()

    def myfun2(sr, end_month=end_month):
        f1 = lambda x: pd.to_datetime(x).year * 12 + pd.to_datetime(x).month
        dif_month = f1(end_month) - f1(sr.name)
        return sr[dif_month] / sr_cum_timing_curve[dif_month] if dif_month < len(sr_cum_timing_curve) else sr.iloc[-1] / \
                                                                                                           sr_cum_timing_curve.iloc[
                                                                                                               -1]

    sr_lifetime_default_rate = df_cum_default_rate.apply(myfun2)

    ### return
    return {'lifetimeDefaultRate': sr_lifetime_default_rate,
            'deltaDefaultAmount': df_delta_default_amount,
            'poolTotalAmount': sr_pool_total_amount,
            'deltaDefaultRate': df_delta_default_rate,
            'defaultTimingCurve': pd.DataFrame({'avg_delta_default_rate': sr_avg_delta_default_rate,
                                                'cum_delta_default_rate': sr_cum_delta_default_rate,
                                                'delta_timing_curve': sr_delta_timing_curve,
                                                'cum_timing_curve': sr_cum_timing_curve}),
            'cumlativeDefaultRate': df_cum_default_rate}


## 异常值检测
def f_detect_outlier(sr_input_values, method='triple'):
    ## 异常值检测函数：三倍标准差、留一
    ## 输入：
    ## 原始值序列（sr_input_values）、方法（method）
    ## 输出：
    ## 数据框包含原始值（input_values）、是否异常值（if_outlier）

    if method == 'triple':
        # triple std (triple)
        mu = np.mean(sr_input_values)
        sigma = np.std(sr_input_values)
        sr_if_outlier = (sr_input_values < (mu - 3 * sigma)) | (sr_input_values > (mu + 3 * sigma))
        return pd.DataFrame({'input_values': sr_input_values, 'if_outlier': sr_if_outlier},
                            columns=['input_values', 'if_outlier'])
    else:
        # leave one out (loo)
        len_loss = len(sr_input_values)
        arr_p_value = np.zeros(len_loss)
        for i in range(len_loss):
            mu1 = np.mean(np.log(sr_input_values.drop(sr_input_values.index[i])))
            sigma1 = np.std(np.log(sr_input_values.drop(sr_input_values.index[i])))
            arr_p_value[i] = 1 - lognorm.cdf(sr_input_values[i], s=sigma1, scale=np.exp(mu1)) if lognorm.cdf(
                sr_input_values[i], s=sigma1, scale=np.exp(mu1)) > 0.5 else lognorm.cdf(sr_input_values[i], s=sigma1,
                                                                                        scale=np.exp(mu1))
        return pd.DataFrame(
            {'input_values': sr_input_values, 'p_value': arr_p_value, 'if_outlier': arr_p_value < 0.001},
            columns=['input_values', 'p_value', 'if_outlier'])


# %% 违约率分析
def f_pd_analysis(df_test, end_month):
    ## 贷款金额、户数
    f1 = lambda df: pd.Series([df['signed_amount'].sum(), df['放款月份'].count()], index=['贷款金额', '贷款户数'])
    df_new_loan = df_test.groupby('lend_request_id')['signed_amount', '放款月份'].first().groupby('放款月份').apply(f1)
    ## M2当期违约金额
    f5 = lambda df: pd.Series([sum(df.principal_balance * df.if_first_default)], index=['M2逾期金额'])
    df_default_loan = df_test.groupby(['放款月份', '报告月份']).apply(f5).reset_index()
    ## 静态池计算
    df_static_pool = pd.merge(df_default_loan, df_new_loan, left_on='放款月份', right_index=True)
    f2 = lambda x: pd.to_datetime(x).year * 12 + pd.to_datetime(x).month
    df_static_pool['周期'] = df_static_pool.报告月份.map(f2) - df_static_pool.放款月份.map(f2)
    dict_result = f_delta_loss_curve_1(df_static_pool, '放款月份', '周期', '贷款金额', 'M2逾期金额', end_month=end_month)
    ## 检测异常值
    sr_lifetime_default_rate = dict_result['lifetimeDefaultRate'][
        dict_result['lifetimeDefaultRate'] > 0].dropna()  # omit NA
    df_if_outlier = f_detect_outlier(sr_lifetime_default_rate, method='triple')
    ## 返回
    # 统计量
    pd_mean = df_if_outlier.input_values[np.logical_not(df_if_outlier.if_outlier)].mean()
    pd_std = df_if_outlier.input_values[np.logical_not(df_if_outlier.if_outlier)].std()
    pd_base = pd_mean
    norm_mean = np.log(pd_mean) - np.power(pd_std, 2) / 2
    norm_std = np.sqrt(np.log(1 + np.power(pd_std / pd_mean, 2)))
    pd_press = lognorm.ppf(q=0.95, s=norm_std, scale=np.exp(norm_mean))
    pd_extreme = lognorm.ppf(q=0.99, s=norm_std, scale=np.exp(norm_mean))
    df_stats = pd.DataFrame(
        {'mean': pd_mean, 'standard': pd_std, 'base': pd_base, 'press': pd_press, 'extreme': pd_extreme}, index=[0],
        columns=['mean', 'standard', 'base', 'press', 'extreme'])
    # 密度曲线
    sr_pd_rate = np.arange(0.01, 0.5, 0.005)
    sr_pd_cdf = lognorm.cdf(sr_pd_rate, s=norm_std, scale=np.exp(norm_mean))
    sr_pd_pdf = lognorm.pdf(sr_pd_rate, s=norm_std, scale=np.exp(norm_mean))
    df_density_curves = pd.DataFrame({'rate': sr_pd_rate, 'cdf': sr_pd_cdf, 'pdf': sr_pd_pdf},
                                     columns=['rate', 'cdf', 'pdf'])
    # 时间分布曲线
    dict_result['defaultTimingCurve'] = dict_result['defaultTimingCurve'].drop(0, errors='ignore')
    sr_pd_timing_initial = com_initial(dict_result['defaultTimingCurve'].index,
                                       dict_result['defaultTimingCurve'].delta_timing_curve)
    sr_pd_timing_optimize = find_optimize(sr_pd_timing_initial, dict_result['defaultTimingCurve'].delta_timing_curve,
                                          dict_result['defaultTimingCurve'].index)
    sr_pd_timing_front = com_initial(dict_result['defaultTimingCurve'].index,
                                     dict_result['defaultTimingCurve'].delta_timing_curve, hold=0.9999)
    df_timing_curves = pd.DataFrame({'term': dict_result['defaultTimingCurve'].index, 'baseRate': sr_pd_timing_optimize,
                                     'frontRate': sr_pd_timing_front}, columns=['term', 'baseRate', 'frontRate'])

    return df_stats.round(6), df_density_curves.round(6), df_timing_curves.round(6)


# %% 早偿率分析
def f_pp_analysis(df_test, end_month):
    ## 计算早偿金额
    df_default_list = df_test.loc[df_test.if_first_default == 1, ['lend_request_id']]
    df_default_list['if_default'] = 1
    df_test = pd.merge(df_test, df_default_list, how='left', on='lend_request_id')
    df_test['早偿金额'] = (df_test['repay_principal'] - df_test['schMnthPrincipal']) * (
        df_test['repay_principal'] > df_test['schMnthPrincipal']) * df_test['if_default'].isnull()
    df_prepay = df_test.groupby(['放款月份', '报告月份'])[['早偿金额']].sum().reset_index()
    ## 贷款金额、笔数
    f1 = lambda df: pd.Series([df['signed_amount'].sum(), df['放款月份'].count()], index=['贷款金额', '贷款户数'])
    df_new_loan = df_test.groupby('lend_request_id')['signed_amount', '放款月份'].first().groupby('放款月份').apply(f1)
    ## 静态池计算
    df_prepay = pd.merge(df_prepay, df_new_loan, left_on='放款月份', right_index=True)
    f2 = lambda x: pd.to_datetime(x).year * 12 + pd.to_datetime(x).month
    df_prepay['周期'] = df_prepay.报告月份.map(f2) - df_prepay.放款月份.map(f2)
    dict_prepay = f_delta_loss_curve_1(df_prepay, '放款月份', '周期', '贷款金额', '早偿金额', end_month=end_month)
    ## 检测异常值
    sr_lifetime_prepay_rate = dict_prepay['lifetimeDefaultRate'][
        dict_prepay['lifetimeDefaultRate'] > 0].dropna()  # omit NA
    df_if_outlier = f_detect_outlier(sr_lifetime_prepay_rate, method='triple')
    ## 返回
    # 统计量
    pp_mean = df_if_outlier.input_values[np.logical_not(df_if_outlier.if_outlier)].mean()
    pp_std = df_if_outlier.input_values[np.logical_not(df_if_outlier.if_outlier)].std()
    pp_base = pp_mean
    a, b = cal_ab2(pp_mean, pp_std)
    pp_press = beta.ppf(q=0.95, a=a, b=b)
    pp_extreme = beta.ppf(q=0.99, a=a, b=b)
    df_stats = pd.DataFrame(
        {'mean': pp_mean, 'standard': pp_std, 'base': pp_base, 'press': pp_press, 'extreme': pp_extreme}, index=[0],
        columns=['mean', 'standard', 'base', 'press', 'extreme'])
    # 密度曲线
    sr_pp_rate = np.arange(0.01, 1, 0.01)
    sr_pp_cdf = beta.cdf(sr_pp_rate, a=a, b=b)
    sr_pp_pdf = beta.pdf(sr_pp_rate, a=a, b=b)
    df_density_curves = pd.DataFrame({'rate': sr_pp_rate, 'cdf': sr_pp_cdf, 'pdf': sr_pp_pdf},
                                     columns=['rate', 'cdf', 'pdf'])
    # 时间分布曲线
    dict_prepay['defaultTimingCurve'] = dict_prepay['defaultTimingCurve'].drop(0, errors='ignore')
    sr_pp_timing_initial = com_initial(dict_prepay['defaultTimingCurve'].index,
                                       dict_prepay['defaultTimingCurve'].delta_timing_curve)
    sr_pp_timing_optimize = find_optimize(sr_pp_timing_initial, dict_prepay['defaultTimingCurve'].delta_timing_curve,
                                          dict_prepay['defaultTimingCurve'].index)
    sr_pp_timing_front = com_initial(dict_prepay['defaultTimingCurve'].index,
                                     dict_prepay['defaultTimingCurve'].delta_timing_curve, hold=0.9999)
    df_timing_curves = pd.DataFrame({'term': dict_prepay['defaultTimingCurve'].index, 'baseRate': sr_pp_timing_optimize,
                                     'frontRate': sr_pp_timing_front}, columns=['term', 'baseRate', 'frontRate'])

    return df_stats.round(6), df_density_curves.round(6), df_timing_curves.round(6)


# %% 回收率分析
def f_pr_analysis(df_test, end_month):
    ## 回收期
    df_recovery = pd.merge(
        df_test.loc[df_test.if_first_default == 1, ['放款月份', 'lend_request_id', 'period', 'phase', 'principal_balance']],
        df_test.loc[:, ['lend_request_id', 'phase', 'status_monthend']], on='lend_request_id')
    df_recovery['回收期'] = df_recovery['phase_y'] - df_recovery['phase_x']
    df_recovery = df_recovery.loc[df_recovery['回收期'] >= 0, :]
    ## 是否首次回收
    df_first_recover = df_recovery.loc[df_recovery.status_monthend < 2, :].groupby(
        'lend_request_id').first().reset_index()
    df_first_recover['if_first_recover'] = 1
    df_recovery = pd.merge(df_recovery, df_first_recover[['lend_request_id', '回收期', 'if_first_recover']], how='left',
                           on=['lend_request_id', '回收期'])
    ## 违约金额
    f3_1 = lambda df: pd.Series([df['principal_balance'].sum(), df['放款月份'].count()], index=['违约金额', '违约户数'])
    df_default_bymonth = df_recovery.groupby('lend_request_id').first().groupby('放款月份').apply(f3_1)
    ## 当期回收金额
    f3_2 = lambda df: pd.Series([sum(df.principal_balance * (df.if_first_recover == 1))], index=['当期回收金额'])
    df_recovery_bymonth = df_recovery.groupby(['放款月份', '回收期']).apply(f3_2).reset_index()
    ## 静态池计算
    df_recovery_static = pd.merge(df_recovery_bymonth, df_default_bymonth, left_on='放款月份', right_index=True)
    dict_recovery = f_delta_loss_curve_1(df_recovery_static, '放款月份', '回收期', '违约金额', '当期回收金额', end_month=end_month)
    ## 检测异常值
    sr_lifetime_recovery_rate = dict_recovery['lifetimeDefaultRate'][
        dict_recovery['lifetimeDefaultRate'] > 0].dropna()  # omit NA
    df_if_outlier = f_detect_outlier(sr_lifetime_recovery_rate, method='triple')
    ## 返回
    # 统计量
    pr_mean = df_if_outlier.input_values[np.logical_not(df_if_outlier.if_outlier)].mean()
    pr_std = df_if_outlier.input_values[np.logical_not(df_if_outlier.if_outlier)].std()
    pr_base = pr_mean
    a, b = cal_ab2(pr_mean, pr_std)
    pr_press = beta.ppf(q=0.95, a=a, b=b)
    pr_extreme = beta.ppf(q=0.99, a=a, b=b)
    df_stats = pd.DataFrame(
        {'mean': pr_mean, 'standard': pr_std, 'base': pr_base, 'press': pr_press, 'extreme': pr_extreme}, index=[0],
        columns=['mean', 'standard', 'base', 'press', 'extreme'])
    # 密度曲线
    sr_pr_rate = np.arange(0.01, 1, 0.01)
    sr_pr_cdf = beta.cdf(sr_pr_rate, a=a, b=b)
    sr_pr_pdf = beta.pdf(sr_pr_rate, a=a, b=b)
    df_density_curves = pd.DataFrame({'rate': sr_pr_rate, 'cdf': sr_pr_cdf, 'pdf': sr_pr_pdf},
                                     columns=['rate', 'cdf', 'pdf'])
    # 时间分布曲线
    dict_recovery['defaultTimingCurve'] = dict_recovery['defaultTimingCurve'].drop(0, errors='ignore')
    #    sr_pr_timing_initial = com_initial(dict_recovery['defaultTimingCurve'].index, dict_recovery['defaultTimingCurve'].delta_timing_curve)
    #    sr_pr_timing_optimize = find_optimize(dict_recovery['defaultTimingCurve'].index, sr_pr_timing_initial, dict_recovery['defaultTimingCurve'].delta_timing_curve)
    sr_pr_timing_back = com_initial(dict_recovery['defaultTimingCurve'].index,
                                    dict_recovery['defaultTimingCurve'].delta_timing_curve, hold=0.9999)
    df_timing_curves = pd.DataFrame({'term': dict_recovery['defaultTimingCurve'].index[0:4],
                                     'baseRate': dict_recovery['defaultTimingCurve'].delta_timing_curve[0:4],
                                     'backRate': sr_pr_timing_back[0:4]}, columns=['term', 'baseRate', 'backRate'])

    return df_stats.round(6), df_density_curves.round(6), df_timing_curves.round(6)


# %%
def timeit(func):
    '''
    装饰器，计算函数执行时间
    '''

    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        exec_time = time_end - time_start
        print('{function} exec time: {time}s'.format(function=func.__name__, time=exec_time))
        return result

    return wrapper


def cal_ab(x):
    '''
    计算alpha、beta值
    '''
    x_mean = np.mean(x)
    x_std = np.std(x)
    alpha_ = x_mean ** 2 * (1 - x_mean) / (x_std ** 2) - x_mean
    beta_ = alpha_ * (1 / x_mean - 1)
    return alpha_, beta_


def cal_ab2(x_mean, x_std):
    '''
    计算alpha、beta值
    '''
    alpha_ = x_mean ** 2 * (1 - x_mean) / (x_std ** 2) - x_mean
    beta_ = alpha_ * (1 / x_mean - 1)
    return alpha_, beta_


def adjust_thre(x, alpha_, beta_):
    '''
    调整阈值：
    基准分布0.99
    前置分布0.9999
    '''
    d = beta.ppf(x, alpha_, beta_)
    return x, d


def beta_distribution(term, alpha_, beta_, param, prr):
    '''
    beta分布
    '''
    ls = []
    term = term.tolist()
    for i in term:
        if i > 1:
            result = beta(alpha_, beta_).cdf(param / len(term) * i) / prr - beta(alpha_, beta_).cdf(
                param / len(term) * (i - 1)) / prr
        else:
            result = beta(alpha_, beta_).cdf(param / len(term) * i) / prr
        ls.append((i, result))

    return ls


@timeit
def find_optimize(pre, real, term):
    """
    违约率优化函数
    """
    loss = mean_squared_error(real, pre)
    real_mean = np.mean(real)
    real_std = np.std(real)
    m_weight = list(np.arange(1, 2, 0.1))
    v_weight = list(np.arange(1, 2, 0.1))
    prr = list(np.arange(0.980, 0.999, 0.001))
    threshold = 5e-5
    st = time.time()
    ls = {}
    while loss > threshold:
        r_mean = random.choice(m_weight) * real_mean
        r_std = random.choice(v_weight) * real_std
        pr = random.choice(prr)
        alp, bet = cal_ab2(r_mean, r_std)
        d = beta(alp, bet).ppf(pr)
        dis = beta_distribution(term, alp, bet, d, pr)
        x, pre = zip(*dis)
        loss = mean_squared_error(real, pre)
        ls[loss] = (alp, bet, d, pr)
        if time.time() - st > 3:
            threshold = 5e-4
        elif len(ls) > 30:
            a, b, c, d = ls.get(min(ls.keys()))
            dis = beta_distribution(term, a, b, c, d)
            x, pre = zip(*dis)
            break

    # print(ls)
    return pre


def com_initial(term, x, hold=0.99):
    a, b = cal_ab(x)
    c, d = adjust_thre(hold, a, b)
    ddd = beta_distribution(term, a, b, d, c)
    x, y = zip(*ddd)
    return y


# %%
if __name__ == '__main__':
    server.run(debug=True, port=1101, host='0.0.0.0')
