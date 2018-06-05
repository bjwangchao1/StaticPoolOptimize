import pandas as pd
import numpy as np
from numba import jit
import datetime


def f_repay_preprocess(df_repay):
    """
    还款分布表预处理：日期转换、计算贷款状态（-1,0,1,2...）和剩余本金
    输入：原始还款分布表（df_repay）
    输出：预处理后的还款分布表（df_repay）
    """
    # to datetime
    df_repay['repaid_time'] = df_repay['repaid_time'].fillna(0).astype('int')
    df_repay[['pass_time', 'due_date', 'repaid_time']] = df_repay[
        ['pass_time', 'due_date', 'repaid_time']].apply(pd.to_datetime, format='%Y%m%d', errors='coerce')
    # # 月份
    df_repay['放款月份'] = df_repay['pass_time'].map(lambda x: x.strftime("%Y-%m"))
    df_repay['报告月份'] = df_repay['due_date'].map(lambda x: x.strftime("%Y-%m"))
    df_repay['实还月份'] = df_repay['repaid_time'].map(lambda x: np.nan if pd.isnull(x) else x.strftime("%Y-%m"))

    # to datetime and add column monthend
    df_repay['monthend'] = df_repay['due_date'] + pd.offsets.MonthBegin(1)
    df_repay['monthend'] = df_repay['monthend'].map(lambda x: x.replace(hour=0, minute=0, second=0))
    df_repay = df_repay.sort_values(by=['cont_no', 'cont_term_no'])

    # 计算逐期的状态、剩余本金
    id = df_repay['lend_request_id'].values
    term = df_repay['phase'].values
    due_date = df_repay['due_date'].values
    temp = df_repay['monthend'].max() + datetime.timedelta(days=1)
    repay_date = df_repay['repaid_time'].fillna(temp).values
    total_principal = df_repay['signed_amount'].values
    due_principal = df_repay['schMnthPrincipal'].values
    repay_principal = df_repay['repay_principal'].values
    monthend = df_repay['monthend'].values

    arr_status_monthend, arr_principal_balance, arr_if_first_default = f_calculate_status_etc(id, term, due_date,
                                                                                              repay_date,
                                                                                              total_principal,
                                                                                              due_principal,
                                                                                              repay_principal, monthend)

    df_repay['status_monthend'] = arr_status_monthend
    df_repay['principal_balance'] = arr_principal_balance
    df_repay['if_first_default'] = arr_if_first_default
    return df_repay


@jit
def f_calculate_status_etc(id, term, due_date, repay_date, total_principal, due_principal, repay_principal, monthend):
    ### calculate the relevant
    arr_if_first_default = np.zeros(len(df_repay))
    arr_status_monthend = np.zeros(len(term))
    arr_principal_balance = np.zeros(len(term))
    current_no = ''

    for i in list(range(len(term))):
        # 当期周期
        current_term = term[i]
        # 历期是否有还款
        sr_if_repaid = monthend[i] > repay_date[(i - current_term + 1):i + 1]
        # print('repaydate', repay_date[(i - current_term + 1):i+1])
        # 剩余本金
        arr_principal_balance[i] = total_principal[i] - sum(
            repay_principal[(i - current_term + 1):i + 1] * sr_if_repaid)
        # print('repay_principal', repay_principal[(i - current_term + 1):i+1])

        # 历期是否全额还款
        sr_if_repaid_full = sr_if_repaid & np.logical_or(
            (repay_principal[(i - current_term + 1):i + 1] >= due_principal[(i - current_term + 1):i + 1]),
            (arr_principal_balance[(i - current_term + 1):(i + 1)] <= 1))
        # 月末状态（早偿状态：-1）
        repay_year = repay_date[i].astype('datetime64[Y]').astype(int) + 1970
        repay_month = repay_date[i].astype('datetime64[M]').astype(int) % 12 + 1
        due_year = due_date[i].astype('datetime64[Y]').astype(int) + 1970
        due_month = due_date[i].astype('datetime64[M]').astype(int) % 12 + 1
        arr_status_monthend[i] = -1 if ((repay_year * 12 + repay_month) < (due_year * 12 + due_month)) & (
            arr_principal_balance[i] <= 1) else current_term - sum(sr_if_repaid_full)
        # print('status',arr_status_monthend[i] )
        if (arr_status_monthend[i] == 2) & (id[i] != current_no):
            arr_if_first_default[i] = 1
            current_no = id[i]

    return arr_status_monthend, arr_principal_balance, arr_if_first_default


def f_calculate_status_etc_2(df_repay, col_contract_no, col_term, col_due_date, col_repay_date, col_total_principal,
                             col_due_principal, col_repay_principal):
    # 计算每期的月末逾期状态、月末剩余本金，以及首次违约的周期（2017-11-25）
    # 输入：
    # “长格式”的还款记录数据框df_repay，
    # 包含的列有合同号（col_contract_no）、周期（col_term）、
    # 应还日期（col_due_date）、实还日期（col_repay_date）、总本金（col_total_principal）、实还本金（col_repay_principal）
    # 输出：
    # 原数据框增加了月末状态（status_monthend）、月末剩余本金（principal_balance）、是否首次违约（if_first_default）

    # to datetime and add column monthend
    df_repay[col_due_date] = pd.to_datetime(df_repay[col_due_date])
    df_repay[col_repay_date] = pd.to_datetime(df_repay[col_repay_date])
    df_repay['monthend'] = df_repay[col_due_date].apply(
        lambda x: pd.to_datetime(str(x.year + 1) + '-' + '01') if x.month == 12 else pd.to_datetime(
            str(x.year) + '-' + str(x.month + 1)))

    # calculate the relevant
    current_no = ''
    arr_if_first_default = np.zeros(len(df_repay))
    arr_status_monthend = np.zeros(len(df_repay))
    arr_principal_balance = np.zeros(len(df_repay))

    idx = df_repay[col_contract_no].values
    term = df_repay[col_term].values
    due_date = df_repay[col_due_date].values
    temp = df_repay['monthend'].max() + datetime.timedelta(days=1)
    repay_date = df_repay[col_repay_date].fillna(temp).values
    total_principal = df_repay[col_total_principal].values
    due_principal = df_repay[col_due_principal].values
    repay_principal = df_repay[col_repay_principal].values
    monthend = df_repay['monthend'].values

    for i in list(range(len(term))):
        # 当期周期
        current_term = term[i]
        # 历期是否有还款
        sr_if_repaid = monthend[i] > repay_date[(i - current_term + 1):i + 1]
        # print('repaydate', repay_date[(i - current_term + 1):i+1])
        # 剩余本金
        arr_principal_balance[i] = total_principal[i] - sum(
            repay_principal[(i - current_term + 1):i + 1] * sr_if_repaid)
        # print('repay_principal', repay_principal[(i - current_term + 1):i+1])

        # 历期是否全额还款
        sr_if_repaid_full = sr_if_repaid & np.logical_or(
            (repay_principal[(i - current_term + 1):i + 1] >= due_principal[(i - current_term + 1):i + 1]),
            (arr_principal_balance[(i - current_term + 1):(i + 1)] <= 1))
        # 月末状态（早偿状态：-1）
        repay_year = repay_date[i].astype('datetime64[Y]').astype(int) + 1970
        repay_month = repay_date[i].astype('datetime64[M]').astype(int) % 12 + 1
        due_year = due_date[i].astype('datetime64[Y]').astype(int) + 1970
        due_month = due_date[i].astype('datetime64[M]').astype(int) % 12 + 1
        arr_status_monthend[i] = 0 if ((repay_year * 12 + repay_month) < (due_year * 12 + due_month)) & (
            arr_principal_balance[i] <= 1) else current_term - sum(sr_if_repaid_full)
        # print('status',arr_status_monthend[i] )
        if (arr_status_monthend[i] == 2) & (idx[i] != current_no):
            arr_if_first_default[i] = 1
            current_no = idx[i]

    df_repay['status_monthend'] = arr_status_monthend
    df_repay['principal_balance'] = arr_principal_balance
    df_repay['if_first_default'] = arr_if_first_default

    return df_repay


if __name__ == '__main__':
    df_repay_history = pd.read_csv(
        r'C:\Users\bjwangchao1\Desktop\京东abs\StaticPoolOptimize\StaticPool_Opt\df_repay_history.csv',
        engine='python', encoding='utf-8')
    # new_repay_data = f_repay_preprocess(df_repay_history)
    df_test = f_calculate_status_etc_2(df_repay_history, 'lend_request_id', 'phase', 'due_date', 'repaid_time',
                                       'signed_amount', 'schMnthPrincipal', 'repay_principal')

    raw_data = pd.read_pickle(r'C:\Users\bjwangchao1\Desktop\京东abs\静态池分析\df_repay.pkl')
