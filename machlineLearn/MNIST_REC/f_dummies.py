# 理解get_dummy的作用
'''
    DataFrame:表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型
    DataFrame 会自动加上索引（跟Series一样），且全部列会被有序排列 （index 代表行 ，columns代表列）
'''

import pandas as pd

df1 = pd.DataFrame([['green', 'A'], ['red', 'B'], ['blue', 'B']])

df1.columns = ['color', 'class']

pd.get_dummies(df1.columns)
print("Output hte table :")
print(df1)
print("Output the special row :")
print(df1.color)

'''
    ix:   新版本不支持，能用数字、index、columns标签进行索引
    iloc：获取行，只能用数字                              >>>iloc[0:1]
    loc ：获取行,列，只能用index的标签                     >>>loc['a'] , loc[:,'a']
'''
df2 = pd.DataFrame({'name': ['hjh', 'csy'], 'sex': ['male', 'female']}, index=['one', 'two'])
print("\nOutput table of df2 :")
print(df2)
print("\nOutput the special row of df2 :")
print(df2.loc['two'])
df2.loc['three'] = ['bob', 'NAN', ]
print("\nOutput changed table of df2 :")
print(df2)
print("\n output the special row of df2 :")
print(df2.loc[:,'name'])
print("用ilocq切片输出第一行")
print(df2.iloc[0:1])