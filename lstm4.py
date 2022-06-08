''' -*- coding = utf-8 -*-'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
import matplotlib.pyplot as plt
import datetime
from pykalman import KalmanFilter
import akshare as ak
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,classification_report
from math import sqrt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']#以黑体的字体显示中文
mpl.rcParams['axes.unicode_minus'] = False#解决保存图像是符号显示为方块的问题

### 一、读取数据

## 读取excel文件，并将‘日期’列解析为日期时间格式,并设为索引
stock_data=pd.read_excel('./002568.xlsx',parse_dates=['日期'],index_col=None)

#对股票数据的列名重新命名



stock_data.columns=['time','open','high','low','Volume','amount','Amplitude','risefall_Amplitude','risefall_amount','turnover','close']
real_data = stock_data  #  备份数据

all_maxstockprice = []
all_minstockprice = []
all_maxtruey = []
all_mintruey = []
all_maxpredy = []
all_minpredy = []
all_mean_truey= []
all_mean_predy= []

stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(symbol="sz002568", start_date="20170111", end_date="20220320", adjust="qfq")
number = stock_zh_a_daily_qfq_df.values[:,-2] # 流通股本数


stock_zh_a_hist_163_df = ak.stock_zh_a_hist_163(symbol="sz002568", start_date="20170111", end_date="20220320")
df=stock_zh_a_hist_163_df 

df = df[~df['成交量'].isin([0])]


total =df['总市值'].values# 总市值
market =df['流通市值'].values# 流通市值
a = total/market # 市值转换因子





observations = stock_data['close'].values
for i in range(1,1001):	
	# 卡尔曼滤波
	def Kalman1D(observations,damping=1):
		# To return the smoothed time series data
		observation_covariance = damping # 观测偏差 （当前数据）
		initial_value_guess = observations[0]
		transition_matrix = 1
		transition_covariance =0.1 #预测偏差 (历史数据)
		initial_value_guess
		kf = KalmanFilter(
				initial_state_mean=initial_value_guess,
				initial_state_covariance=observation_covariance,
				observation_covariance=observation_covariance,
				transition_covariance=transition_covariance,
				transition_matrices=transition_matrix
			)
		pred_state, state_cov = kf.smooth(observations)

		return pred_state
		
	new= Kalman1D(observations,1.5)

	x= stock_data['time']


	# ### 1.原始值与经过滤波后的数值对比
	# plt.plot(x,observations,color='red',label='原始真实值',)
	# plt.plot(x,new ,color='black',label='滤波后值',linestyle='--')
	# plt.xlabel('时间')
	# plt.ylabel('总市值')
	# plt.title('2011.3—2022.03')
	# plt.legend()
	# plt.show()



	stock_data["after_kalman_close"] = new







	# for i in range(1,30):
	# 	stock_data["previous_days_stock_price%.6f"%i] = stock_data["close"].shift(-i) # 滞后n期

	stock_data["previous_days_stock_price5"] = stock_data["after_kalman_close"].shift(-5) # 滞后五期
	



	## 划分训练集
	start = datetime.datetime(2011, 1, 1, 0, 0, 0)
	end = datetime.datetime(2017,1,1,0,0,0)
	subset = stock_data[stock_data['time']>start]
	train = subset[subset['time']<end]  

	## 划分测试集
	start = datetime.datetime(2017, 1, 1, 0, 0, 0)
	end = datetime.datetime(2022,3,20,0,0,0)
	subset = stock_data[stock_data['time']>start]
	test = subset[subset['time']<end]


	x= train['time']
	train_2 = train['close']
	ob2 = train['after_kalman_close']

	# ### 2.训练集原始值与经过滤波后的数值对比
	# plt.plot(x,train_2,color='red',label='训练集原始真实值',)
	# plt.plot(x,ob2,color='black',label='训练集滤波后值',linestyle='--')
	# plt.xlabel('时间')
	# plt.ylabel('总市值')
	# plt.title('2011.3—2016.12')
	# plt.legend()
	# plt.show()



	x= test['time']
	test_2 = test['close']
	ob3 = test['after_kalman_close']

	# ### 3.测试集原始值与经过滤波后的数值对比
	# plt.plot(x,test_2,color='red',label='测试集原始真实值',)
	# plt.plot(x,ob3,color='black',label='测试集滤波后值',linestyle='--')
	# plt.xlabel('时间')
	# plt.ylabel('总市值')
	# plt.title('2017.1—2022.03')
	# plt.legend()
	# plt.show()




	### 二、股票数据预处理

	## 训练集处理
	#获取DataFrame中的数据，形式为数组array形式

	train = train.values[:,1:]
	train_values=train

	#确保所有数据为float类型
	train_values=train_values.astype('float32')

	# 特征的归一化处理
	scaler = MinMaxScaler(feature_range=(0, 1))
	train_scaled = scaler.fit_transform(train_values)


	## 测试集处理
	#获取DataFrame中的数据，形式为数组array形式
	test = test.values[:,1:]
	test_values=test
	#确保所有数据为float类型
	test_values=test_values.astype('float32')

	# 特征的归一化处理
	scaler = MinMaxScaler(feature_range=(0, 1))
	test_scaled = scaler.fit_transform(test_values)




	# 2.将数据集转化为监督学习状态
	#定义series_to_supervised()函数
	#将时间序列转换为监督学习问题
	def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
		
		# Frame a time series as a supervised learning dataset.
		# Arguments:
		# data：输入数据需要是列表或二维的NumPy数组的观察序列。
		# n_in：输入的滞后观察数（X）。值可以在[1..len（data）]之间，可选的。默认为1。
		# n_out：输出的观察数（y）。值可以在[0..len（data）-1]之间，可选的。默认为1。
		# dropnan：Bool值，是否删除具有NaN值的行，可选的。默认为True
		
		n_vars = 1 if type(data) is list else data.shape[1]
		df = pd.DataFrame(data)
		cols, names = list(), list() # 创建空列表
		# input sequence (t-n, ... t-1)
		for i in range(n_in, 0, -1):
			cols.append(df.shift(i))
			names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		# forecast sequence (t, t+1, ... t+n)
		for i in range(0, n_out):
			cols.append(df.shift(-i))
			if i == 0:
				names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
			else:
				names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
		# put it all together
		agg = pd.concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values
		if dropnan:
			agg.dropna(inplace=True)
		return agg

	#将时间序列转换为监督学习问题
	train_reframed = series_to_supervised(train_scaled, 1, 1)
	test_reframed = series_to_supervised(test_scaled, 1, 1)





	## 四 将数据划分为训练集和测试集后的转换

	# 划分训练集和测试集  待改进，按时间划分

	train = train_reframed.values
	test = test_reframed.values



	# 划分训练集和测试集的输入和输出
	train_X, train_y = train[:, :-1], train[:, -1]
	test_X, test_y = test[:, :-1], test[:, -1]




	#转化为三维数据
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	# print(train_X.shape, train_y.shape)
	# print(test_X.shape, test_y.shape)




## 五 模型构建及其预测
# 1.
# 搭建LSTM模型

	model = Sequential()
	model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dropout(0.5))
	model.add(Dense(1,activation='relu'))
	model.compile(loss='mae', optimizer='adam')
	
	history = model.fit(train_X, train_y, epochs=40, batch_size=90, validation_data=(test_X, test_y), verbose=2,shuffle=False)  #原始ep50，bsize =100
	
	# #4.绘制损失图
	# plt.plot(history.history['loss'], label='train')
	# plt.plot(history.history['val_loss'], label='test')
	# plt.title('LSTM_0002568.SH', fontsize='12')
	# plt.ylabel('loss', fontsize='10')
	# plt.xlabel('epoch', fontsize='10')
	# plt.legend()
	# plt.show()

	# 2.
	#模型预测收益率
	y_predict = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
	

	#将预测结果按比例反归一化
	inv_y_test = np.concatenate((test_X[:, :11],y_predict), axis=1)
	inv_y_test = scaler.inverse_transform(inv_y_test)
	inv_y_predict=inv_y_test[:,-1]


	#将真实结果按比例反归一化
	test_y = test_y.reshape((len(test_y), 1))
	inv_y_train = np.concatenate((test_X[:, :11],test_y), axis=1)
	inv_y_train = scaler.inverse_transform(inv_y_train)
	inv_y = inv_y_train[:, -1]
	# print('反归一化后的预测结果：',inv_y_predict)
	# print('反归一化后的真实结果：',inv_y)



	# plt.plot(inv_y,color='red',label='滤波后的真实值')
	# plt.plot(inv_y_predict,color='grey',label='滤波后的预测',linestyle='--')
	# plt.xlabel('时间')
	# plt.ylabel('值')
	# plt.title('2017.1—2022.03')
	# plt.legend()
	# plt.show()





	## 六 模型评估

	
	#回归评价指标
	# calculate MSE 均方误差
	# mse=mean_squared_error(inv_y,inv_y_predict)
	# # calculate RMSE 均方根误差
	# rmse = sqrt(mean_squared_error(inv_y, inv_y_predict))
	# #calculate MAE 平均绝对误差
	# mae=mean_absolute_error(inv_y,inv_y_predict)
	# #calculate R square
	# r_square=r2_score(inv_y,inv_y_predict)
	# print('均方误差: %.6f' % mse)
	# print('均方根误差: %.6f' % rmse)
	# print('平均绝对误差: %.6f' % mae)
	# print('R_square: %.6f' % r_square)
	# print(inv_y[-8:])

	# # print(inv_y_predict[-8:])
	# stock_data = stock_data[-1260:]
	# stock_data["inv_y_predict"] = inv_y_predict


	# stock_data.to_excel('./1.xlsx',index=None,encoding='utf-8') 
	# ## 七.市值







	## 划分测试集
	paintdata = real_data



	start = datetime.datetime(2017,1,10, 0, 0, 0)
	end = datetime.datetime(2022,3,20,0,0,0)
	subset1 = real_data[real_data['time']>start]
	paintdata = subset1[subset1['time']<end]



	paintdata["share_number"] =number
	paintdata["transformkey"] =a
	paintdata["inv_y_predict"] = inv_y_predict


	number = paintdata["share_number"] 
	a = paintdata["transformkey"]
	inv_y_predict = paintdata["inv_y_predict"]
	inv_y = paintdata["close"]


	truey = inv_y*number*a
	predy = inv_y_predict*number*a


	
	mean_truey = np.mean(inv_y)
	# print('滤波后预测市值最大值: %.6f万元' % maxpredy)
	mean_predy = np.mean(inv_y_predict)
	# print('滤波后预测市值最小值: %.6f万元' % minpredy)
	all_mean_truey.append(mean_truey)
	all_mean_predy.append(mean_predy)

	# 5. 绘图
	# x= paintdata['time']
	# plt.plot(x,truey,color='red',label='真实市值',)
	# plt.plot(x,predy ,color='black',label='滤波后预测市值',linestyle='--')
	# plt.xlabel('时间')
	# plt.ylabel('总市值')
	# plt.title('2017.1—2022.03')
	# plt.legend()
	# plt.show()

	maxtruey = max(truey)/10000
	# print('实际市值最大值: %.6f万元' % maxtruey)
	mintruey = min(truey)/10000
	# print('实际市值最小值: %.6f万元' % mintruey)


	maxpredy = max(predy)/10000
	# print('滤波后预测市值最大值: %.6f万元' % maxpredy)
	minpredy = min(predy)/10000
	# print('滤波后预测市值最小值: %.6f万元' % minpredy)

	max_inv_y_predict = max(inv_y_predict)
	min_inv_y_predict = min(inv_y_predict)



	all_maxstockprice.append(max_inv_y_predict)
	all_minstockprice.append(min_inv_y_predict)
	all_maxtruey.append(maxtruey)
	all_mintruey.append(mintruey)
	all_maxpredy.append(maxpredy)
	all_minpredy.append(minpredy)
	print('第 %.6f次' % i)

# 求预测的市值的均值
mean_maxpredy = np.mean(all_maxpredy)
mean_minpredy = np.mean(all_minpredy)
# 求预测股价的均值
mean_maxstockprice = np.mean(all_maxstockprice)
mean_minstockprice = np.mean(all_minstockprice)

# 区间内股价的均值
mean__all_mean_truey= np.mean(all_mean_truey)
mean__all_mean_predy= np.mean(all_mean_predy)


maxtruey = np.mean(all_maxtruey)
mintruey= np.mean(all_mintruey)

maxtruey = max(truey)/10000
print('实际市值最大值: %.6f万元' % maxtruey)
mintruey = min(truey)/10000
print('实际市值最小值: %.6f万元' % mintruey)


#求方差
a_var_max = np.var(all_maxpredy)
a_var_min = np.var(all_minpredy)
a_var_maxprice = np.var(all_maxstockprice)
a_var_minprice = np.var(all_minstockprice)

a_var_mean__all_mean_truey = np.var(all_mean_truey)
a_var_mean__all_mean_predy = np.var(all_mean_predy)

#求标准差
a_std_max = np.std(all_maxpredy,ddof=1)
a_std_min = np.std(all_minpredy,ddof=1)

a_std_maxprice = np.std(all_maxstockprice,ddof=1)
a_std_minprice = np.std(all_minstockprice,ddof=1)

a_std_mean__all_mean_truey = np.std(all_mean_truey,ddof=1)
a_std_mean__all_mean_predy = np.std(all_mean_predy,ddof=1)

print('='*100)



print('='*100)
print('滤波后预测市值最大值的均值: %.6f万元' % mean_maxpredy)
print('滤波后预测市值最小值的均值: %.6f万元' % mean_minpredy)
print('='*100)
print("滤波后预测最大市值均值的方差为：%f" % a_var_max)
print("滤波后预测最小市值均值的方差为：%f" % a_var_min)
print("滤波后预测最大市值均值的标准差为:%f" % a_std_max)
print("滤波后预测最小市值均值的标准差为:%f" % a_std_min)

print('='*100)

maxiy = max(inv_y)
print('实际收盘价最大值: %.6f元' % maxiy)
miniy = min(inv_y)
print('实际收盘价最小值: %.6f元' % miniy)

print('='*100)
print('滤波后预测收盘价最大均值: %.6f' % mean_maxstockprice)
print('滤波后预测收盘价最小均值: %.6f' % mean_minstockprice)
print('='*100)
print("滤波后预测最大收盘价均值的方差为：%f" % a_var_maxprice)
print("滤波后预测最小收盘价均值的方差为：%f" % a_var_minprice)
print("滤波后预测最大收盘价均值的标准差为:%f" % a_std_maxprice)
print("滤波后预测最小收盘价均值的标准差为:%f" % a_std_minprice)
print('='*100)

print('实际收盘价均值: %.6f' % mean__all_mean_truey)
print('滤波后预测收盘价均值: %.6f' % mean__all_mean_predy)
print('='*100)

print("实际收盘价均值的方差为：%f" % a_var_mean__all_mean_truey)
print("滤波后预测收盘价均值的方差为：%f" % a_var_mean__all_mean_predy)
print('='*100)
print("实际收盘价均值的标准差为:%f" % a_std_mean__all_mean_truey)
print("滤波后预测收盘价均值的标准差为:%f" % a_std_mean__all_mean_predy)
## 仿真

