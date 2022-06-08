#@------------------------- CAPM模型计算 -----------------------------------------------------
#环境&数据准备
import sys as sy
import numpy as np
import pandas as pd
import pyecharts as pye
from sklearn import datasets as ds
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pyecharts as pye
import akshare as ak

#Stock_p <- md_stock(c("002568","^000300"), date_range = "5y")
bairun = pd.read_csv("C:\\Users\\13407\\Desktop\\基金管理案例大赛\\bairun.csv")
shenzhen = pd.read_csv("C:\\Users\\13407\\Desktop\\基金管理案例大赛\\shenzheng.csv")

#数据准备
df_tmp1 = pd.DataFrame()
df_tmp2 = pd.DataFrame()
df_tmp1['rt_bairun'] =(bairun.close.diff(1))/bairun.close.shift(1)
df_tmp2['rt_sz'] =(shenzhen.close.diff(1))/shenzhen.close.shift(1)
df_tmp = df_tmp1.join(df_tmp2)
df_tmp = df_tmp.dropna()
del df_tmp1
del df_tmp2
 
 
#计算Beta系数
cov_sm = np.cov(df_tmp.rt_bairun, df_tmp.rt_sz)[0,1]
var_m = np.var(df_tmp.rt_sz)
Beta = cov_sm/var_m
 
Erm = df_tmp.rt_sz.mean()*365  #计算年化市场期望收益率
Rf = 0.015
 
#根据公司Ers = rf + Beta*(Erm - rf)
Ers = Rf + Beta*(Erm - Rf)
 
print('Bata = ' + str(Beta))
print('Rf = ' + str(Rf))
print('Erm = ' + str(Erm))
print('Ers = ' + str(Ers))















