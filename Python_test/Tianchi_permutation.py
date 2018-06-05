import itertools
import pandas as pd

a = [259, 300, 320, 410, 450, 450, 280, 110, 260, 320]
total = pd.DataFrame(list(itertools.permutations(a, 10)))
# 第一种红包分布方法中各个红包是否大于1100，1个赏金的红包不可能超过
total.loc[(total[0] + total[1] + total[2]) > 1100, "type1_1"] = 1
total.loc[(total[3] + total[4] + total[5]) > 1100, "type1_2"] = 1
total.loc[(total[6] + total[7] + total[8]) > 1100, "type1_3"] = 1
# 第二种红包分布方法中各个红包是否大于1100，1个、2个赏金的红包不可能超过
total.loc[(total[0] + total[1] + total[2]) > 1100, "type2_1"] = 1
total.loc[(total[3] + total[4] + total[5]) > 1100, "type2_2"] = 1
total = total.fillna(0)
# 第一种红包分布超过1100的数目：3,3,3,1
total["first_type"] = total["type1_1"] + total["type1_2"] + total["type1_3"]
# 第一种红包分布超过1100的数目：3,3,2,2
total['second_type'] = total["type2_1"] + total["type2_2"]
# 有大红包的概率
P_has_any = (len(total[total["first_type"] > 0]) + len(total[total["second_type"] > 0])) / (len(total) * 2)
# 有1个大红包的概率
P_has_one = (len(total[total["first_type"] == 1]) + len(total[total["second_type"] == 1])) / (len(total) * 2)
# 有2个大红包的概率
P_has_two = (len(total[total["first_type"] == 2]) + len(total[total["second_type"] == 2])) / (len(total) * 2)
# 有3个大红包的概率
P_has_three = (len(total[total["first_type"] == 3]) + len(total[total["second_type"] == 3])) / (len(total) * 2)
print("P_has_any:", P_has_any * 100, "%")
print("P_has_one:", P_has_one * 100, "%")
print("P_has_two:", P_has_two * 100, "%")
print("P_has_three:", P_has_three * 100, "%")
