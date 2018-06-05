import pandas as pd
import numpy as np
import datetime


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
        arr_status_monthend[i] = -1 if ((repay_year * 12 + repay_month) < (due_year * 12 + due_month)) & (
            arr_principal_balance[i] <= 1) else current_term - sum(sr_if_repaid_full)
        # print('status',arr_status_monthend[i] )
        if (arr_status_monthend[i] == 2) & (idx[i] != current_no):
            arr_if_first_default[i] = 1
            current_no = idx[i]

    df_repay['status_monthend'] = arr_status_monthend
    df_repay['principal_balance'] = arr_principal_balance
    df_repay['if_first_default'] = arr_if_first_default

    return df_repay


# def f_calculate_status_etc_2(df_repay, col_contract_no, col_term, col_due_date, col_repay_date, col_total_principal,
#                             col_due_principal, col_repay_principal):
#    # 计算每期的月末逾期状态、月末剩余本金，以及首次违约的周期（2017-11-25）
#    # 输入：
#    # “长格式”的还款记录数据框df_repay，
#    # 包含的列有合同号（col_contract_no）、周期（col_term）、
#    # 应还日期（col_due_date）、实还日期（col_repay_date）、总本金（col_total_principal）、实还本金（col_repay_principal）
#    # 输出：
#    # 原数据框增加了月末状态（status_monthend）、月末剩余本金（principal_balance）、是否首次违约（if_first_default）
#
#    ### to datetime and add column monthend
#    df_repay[col_due_date] = pd.to_datetime(df_repay[col_due_date])
#    df_repay[col_repay_date] = pd.to_datetime(df_repay[col_repay_date])
#    df_repay['monthend'] = df_repay[col_due_date].apply(
#        lambda x: pd.to_datetime(str(x.year + 1) + '-' + '01') if x.month == 12 else pd.to_datetime(
#            str(x.year) + '-' + str(x.month + 1)))
#
#    ### calculate the relevant
#    current_no = ''
#    arr_if_first_default = np.zeros(len(df_repay))
#    arr_status_monthend = np.zeros(len(df_repay))
#    arr_principal_balance = np.zeros(len(df_repay))
#    for i in range(len(df_repay)):
#        # 当期周期
#        current_term = df_repay[col_term].iloc[i]
#        # 历期是否有还款
#        sr_if_repaid = df_repay['monthend'].iloc[i] > df_repay[col_repay_date].iloc[(i - current_term + 1):(i + 1)]
#        # 剩余本金
#        arr_principal_balance[i] = df_repay[col_total_principal].iloc[i] - sum(
#            df_repay[col_repay_principal].iloc[(i - current_term + 1):(i + 1)] * sr_if_repaid)
#        # 历期是否全额还款（部分早偿待处理）
#        sr_if_repaid_full = sr_if_repaid & np.logical_or((df_repay[col_repay_principal].iloc[
#                                                          (i - current_term + 1):(i + 1)] >= df_repay[
#                                                                                                 col_due_principal].iloc[
#                                                                                             (
#                                                                                                 i - current_term + 1):(
#                                                                                                 i + 1)]),
#                                                         (arr_principal_balance[
#                                                          (i - current_term + 1):(i + 1)] <= 1))
#        # 月末状态
#        arr_status_monthend[i] = current_term - sum(sr_if_repaid_full)
#        # 是否首次违约
#        if (arr_status_monthend[i] == 2) & (df_repay[col_contract_no].iloc[i] != current_no):
#            arr_if_first_default[i] = 1
#            current_no = df_repay[col_contract_no].iloc[i]
#
#    df_repay['status_monthend'] = arr_status_monthend
#    df_repay['principal_balance'] = arr_principal_balance
#    df_repay['if_first_default'] = arr_if_first_default
#
#    ### return
#    return (df_repay)


def f_delta_loss_curve_1(df_static_pool, col_pool_month, col_term, col_pool_total, col_delta_default, end_month):
    # 静态池计算方法（delta loss curve method） (2017-11-23)
    # 输入：
    # 长格式的静态池数据框（df_static_pool），
    # 包含列放款月（col_pool_month）、周期（col_term）、月放款总额（col_pool_total）、当期违约金额（col_delta_default）
    # 输出：
    # 到期违约率，以及中间结果

    # calculate the delta default rate
    def myfun1(sr, end_month=end_month):
        f1 = lambda x: pd.to_datetime(x).year * 12 + pd.to_datetime(x).month
        dif_month = f1(end_month) - f1(sr.name)
        sr[:dif_month] = sr[:dif_month].fillna(0)
        return sr

    df_delta_default_amount = df_static_pool.pivot(col_term, col_pool_month, col_delta_default).apply(myfun1)
    sr_pool_total_amount = df_static_pool.groupby(col_pool_month)[col_pool_total].first()
    df_delta_default_rate = df_delta_default_amount / sr_pool_total_amount

    # calculate the expected lifetime default rate
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


# 违约率分析
f1 = lambda df: pd.Series([df['signed_amount'].sum(), df['放款月份'].count()], index=['贷款金额', '贷款户数'])
df_new_loan = data.groupby('lend_request_id')['signed_amount', '放款月份'].first().groupby('放款月份').apply(f1)
# M2当期违约金额
f5 = lambda df: pd.Series([sum(df.principal_balance * df.if_first_default)], index=['M2逾期金额'])
df_default_loan = data.groupby(['放款月份', '报告月份']).apply(f5).reset_index()
# 静态池计算
df_static_pool = pd.merge(df_default_loan, df_new_loan, left_on='放款月份', right_index=True)
f2 = lambda x: pd.to_datetime(x).year * 12 + pd.to_datetime(x).month
df_static_pool['周期'] = df_static_pool.报告月份.map(f2) - df_static_pool.放款月份.map(f2)
dict_result = f_delta_loss_curve_1(df_static_pool, '放款月份', '周期', '贷款金额', 'M2逾期金额', end_month='2018-02')

save_result(dict_result,path='违约率分析_660.xlsx')
# 早偿率分析
df_test = data
# 计算早偿金额
df_default_list = df_test.loc[df_test.if_first_default == 1, ['lend_request_id']]
df_default_list['if_default'] = 1
df_test = pd.merge(df_test, df_default_list, how='left', on='lend_request_id')
df_test['早偿金额'] = (df_test['repay_principal'] - df_test['schMnthPrincipal']) * (
    df_test['repay_principal'] > df_test['schMnthPrincipal']) * df_test['if_default'].isnull()
df_prepay = df_test.groupby(['放款月份', '报告月份'])[['早偿金额']].sum().reset_index()
# 贷款金额、笔数
f1 = lambda df: pd.Series([df['signed_amount'].sum(), df['放款月份'].count()], index=['贷款金额', '贷款户数'])
df_new_loan = df_test.groupby('lend_request_id')['signed_amount', '放款月份'].first().groupby('放款月份').apply(f1)
# 静态池计算
df_prepay = pd.merge(df_prepay, df_new_loan, left_on='放款月份', right_index=True)
f2 = lambda x: pd.to_datetime(x).year * 12 + pd.to_datetime(x).month
df_prepay['周期'] = df_prepay.报告月份.map(f2) - df_prepay.放款月份.map(f2)
dict_prepay = f_delta_loss_curve_1(df_prepay, '放款月份', '周期', '贷款金额', '早偿金额', end_month='2018-02')

save_result(dict_prepay,'早偿率分析_660.xlsx')
# 回收率分析
## 回收期
df_recovery = pd.merge(
    df_test.loc[df_test.if_first_default == 1, ['放款月份', 'lend_request_id', 'period', 'phase', 'principal_balance']],
    df_test.loc[:, ['lend_request_id', 'phase', 'status_monthend']], on='lend_request_id')
df_recovery['回收期'] = df_recovery['phase_y'] - df_recovery['phase_x']
df_recovery = df_recovery.loc[df_recovery['回收期'] > 0, :]
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
dict_recovery = f_delta_loss_curve_1(df_recovery_static, '放款月份', '回收期', '违约金额', '当期回收金额', end_month='2018-02')
save_result(dict_recovery,path='回收率分析_660.xlsx')

def save_result(dict_result1,path):
    writer = pd.ExcelWriter(path)
    # pd.Series([dict_result[x] for x in ['defaultRateMeanNo0','defaultRateStdNo0']], index=['defaultRateMeanNo0','defaultRateStdNo0']).to_excel(writer, sheet_name='DefaultRateMeanStdNo0')
    dict_result1['lifetimeDefaultRate'].to_excel(writer, sheet_name='lifetimeDefaultRate')
    dict_result1['deltaDefaultAmount'].to_excel(writer, sheet_name='deltaDefaultAmount')
    dict_result1['poolTotalAmount'].to_excel(writer, sheet_name='poolTotalAmount')
    dict_result1['deltaDefaultRate'].to_excel(writer, sheet_name='deltaDefaultRate')
    dict_result1['defaultTimingCurve'].to_excel(writer, sheet_name='defaultTimingCurve')
    dict_result1['cumlativeDefaultRate'].to_excel(writer, sheet_name='cumlativeDefaultRate')
    writer.save()


if __name__ == '__main__':
    data = pd.read_csv(r'C:\Users\bjwangchao1\Desktop\凡普医美\df_repay_his0322.txt', engine='python', sep='\t')
    data['放款日期'] = pd.to_datetime(data['放款日期'])
    data['应还日期'] = pd.to_datetime(data['应还日期'])
    data['实还日期'] = pd.to_datetime(data['实还日期'])
    #    data.loc[data.放款日期 < data.应还日期, 'pass_mth'] = data.loc[data.放款日期 < data.应还日期, '放款日期']
    #    data.loc[data.放款日期 >= data.应还日期, 'pass_mth'] = data.loc[data.放款日期 >= data.应还日期, '应还日期'].map(
    #        lambda x: x - timedelta(days=30))

    data_pa = data.groupby('进件号', as_index=False)['应还日期'].first()
    #    data_pa['pass_mth']=data_pa.应还日期.map(lambda x: x - timedelta(days=30))
    data_pa['pass_mth'] = data_pa['应还日期'].apply(
        lambda x: pd.to_datetime(str(x.year - 1) + '-' + '12') if x.month == 1 else pd.to_datetime(
            str(x.year) + '-' + str(x.month - 1)))
    data = pd.merge(data, data_pa[['进件号', 'pass_mth']], on='进件号', how='left')

    data['放款月份'] = data['pass_mth'].map(lambda x: x.strftime("%Y-%m"))
    data['报告月份'] = data['应还日期'].map(lambda x: x.strftime("%Y-%m"))
    data['实还月份'] = data['实还日期'].map(lambda x: np.nan if pd.isnull(x) else x.strftime("%Y-%m"))
    latest_repaid_time = data.实还日期.max()
    data = data.loc[data.应还日期 < pd.to_datetime(latest_repaid_time)]
    df = f_calculate_status_etc_2(data, '进件号', '期数', '应还日期', '实还日期', '合同金额', '应还本金', '实还本金')

    data.rename(columns={'进件号': 'lend_request_id', '实还本金': 'repay_principal', '应还本金': 'schMnthPrincipal'}, inplace=True)
    data.rename(columns={'合同金额': 'signed_amount'}, inplace=True)
    data.rename(columns={'期数': 'phase', '总期数': 'period'},inplace=True)


    df_score=pd.read_csv(r'C:\Users\bjwangchao1\Desktop\finup_ecmm\f_addr_data\f_mix_model\fp_f_score36705.csv')
    df_score_660 = df_score.loc[df_score.f_score >= 660]
    df_score_670 = df_score.loc[df_score.f_score >= 670]
    df_score_680 = df_score.loc[df_score.f_score >= 680]
    data=pd.merge(df_score_680[['lend_request_id']],data,on='lend_request_id',how='inner')
    # df1 = df.loc[df.放款日期 < pd.to_datetime('2017-11')]
    #    df112 = pd.read_csv(r'112.csv', encoding='gbk')
    #    df6 = pd.read_csv(r'66.csv', encoding='gbk')
    #
    #
    #    def qianlong_status(data):
    #        trans_mat = np.zeros((10, 11))
    #        for i in range(data.shape[0]):
    #            for j in range(data.shape[1] - 1):
    #                for k in range(10):
    #                    if k < 9:
    #                        if (data.values[i, j] == k) & (pd.notnull(data.values[i, j + 1])):
    #                            trans_mat[k, 10] += 1
    #                            for m in range(k + 2):
    #                                if data.values[i, j + 1] == m:
    #                                    trans_mat[k, m] += 1
    #                                else:
    #                                    trans_mat[k, m] += 0
    #                        else:
    #                            trans_mat[k, 10] += 0
    #                    else:
    #                        if (data.values[i, j] == k) & (pd.notnull(data.values[i, j + 1])):
    #                            trans_mat[k, 10] += 1
    #                            trans_mat[k, 9] += 1
    #        trans_df = pd.DataFrame(trans_mat, index=['mo', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9'],
    #                                columns=['mo', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'total'])
    #        return trans_df
    #
    #
    #    def ym_status_6(data):
    #        trans_mat = np.zeros((7, 8))
    #        for i in range(data.shape[0]):
    #            for j in range(data.shape[1] - 1):
    #                for k in range(7):
    #                    if k < 6:
    #                        if (data.values[i, j] == k) & (pd.notnull(data.values[i, j + 1])):
    #                            trans_mat[k, 7] += 1
    #                            for m in range(k + 2):
    #                                if data.values[i, j + 1] == m:
    #                                    trans_mat[k, m] += 1
    #                                else:
    #                                    trans_mat[k, m] += 0
    #                        else:
    #                            trans_mat[k, 7] += 0
    #                    else:
    #                        if (data.values[i, j] == k) & (pd.notnull(data.values[i, j + 1])):
    #                            trans_mat[k, 7] += 1
    #                            trans_mat[k, 6] += 1
    #        trans_df = pd.DataFrame(trans_mat, index=['mo', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6'],
    #                                columns=['mo', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'total'])
    #        return trans_df
    #
    #
    #    dft2 = ym_status_6(df6)
    #
    #
    #    def transform_x(data):
    #        ddr = data.groupby('进件号')['status_monthend'].apply(lambda x: str(list(x)).strip('[]')).reset_index()
    #        ddr_2 = ddr.status_monthend.str.split(',', expand=True)
    #        ddr_2.columns = range(1, 10)
    #        ddr_2[0] = 0
    #        ddr_2 = ddr_2.applymap(np.float64)
    #        ddr_2.sort_index(axis=1, inplace=True)
    #        return ddr_2
