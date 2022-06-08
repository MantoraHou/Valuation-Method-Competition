#@------------------------------------------ 淘宝数据爬取 -------------------------------------
##1.从淘宝中获取数据
from selenium import webdriver
import time
import re
import csv
from selenium.webdriver.common.by import By
import json
#输入框，搜索按钮
def search_product():
    driver.find_element(By.XPATH,'//*[@id="q"]').send_keys(kw)
    driver.find_element(By.XPATH,'//*[@id="J_TSearchForm"]/div[1]/button').click()
    # 扫码登录
    time.sleep(10)
    #得到查询的总页数
    token=driver.find_element(By.XPATH,'//*[@id="mainsrp-pager"]/div/div/div/div[1]').text
    token=int(re.compile('(\d+)').search(token).group(1))
    return token
 
#页面滚动
def drop_down():
        js = "document.documentElement.scrollTop=10000"  # 下拉加载
        driver.execute_script(js)
        time.sleep(2)
 
 
#获得商品数据
def get_product():
    #//代表任意位置
    divs = driver.find_elements(By.XPATH,'//div[@class="items"]/div[@class="item J_MouserOnverReq  "]')
    for div in divs:
       # div.find_element_by_xpath('.//a')被淘汰了
       #产品名字
       goods = div.find_element(By.XPATH,'.//div[@class="row row-2 title"]').text
       #价格
       price = div.find_element(By.XPATH,'.//div[@class="price g_price g_price-highlight"]/strong').text + '元'
       #产品图片
       image = div.find_element(By.XPATH,'.//div[@class="pic"]/a/img').get_attribute('src')
       #交易量
       deal = div.find_element(By.XPATH,'.//div[@class="deal-cnt"]').text
       #店家名字
       name = div.find_element(By.XPATH,'.//div[@class="shop"]/a/span[2]').text
       #店家位置
       location = div.find_element(By.XPATH,'.//div[@class="location"]').text
       
       data_dict = {}  # 定义一个字典存储数据
       data_dict["goods"] = goods
       data_dict["price"] = price
       data_dict["image"] = image
       data_dict["deal"] = deal
       data_dict["name"] = name
       data_dict["location"] = location
       print(data_dict)  # 输出这个字典
       data_list.append(data_dict)  # 将数据存入全局变量中
 
 
#主函数
def next_page():
       token= 46
       drop_down()
       get_product()
       num=1
       while num != token:
           print('--------------------------------------------------------------')
           print("第%s页:" % str(num + 1))
           driver.get('https://s.taobao.com/search?q={}&s={}'.format(kw, 44*num))#可以防止反扒
           time.sleep(20)
           #智能等待 最高等待时间为10s 如果超出10s 抛出异常
           #无限循环进入网页 可能会造成网页卡顿
           driver.implicitly_wait(20)
           drop_down()
           get_product()
           num += 1
 

#存取数据
def save():
    with open('C:\\Users\\13407\\Desktop\\基金管理案例大赛\\taobao.csv','w', encoding='utf-8') as f:
        # 表头
        title = data_list[0].keys()
        # 声明writer
        writer = csv.DictWriter(f, title)
        # 写入表头
        writer.writeheader()
        # 批量写入数据
        writer.writerows(data_list)

if __name__ =='__main__':
    data_list = []  # 设置全局变量来存储数据
    kw=input('请输入你想要查询的商品：')
    #浏览器
    driver= webdriver.Chrome()
    #最大化窗口
    driver.maximize_window()
    driver.get('https://www.taobao.com/')
    next_page()
    save()
#------------------------------------------------------------------------------------------------
##2.数据分析
from pandas.core.frame import DataFrame
data = DataFrame(data_list)
data = data[['location','goods','price','deal']]
data = data.rename(columns={'location':'item_loc','goods':'raw_title','price':'view_price','deal':'view_sales'})
half_count = len(datatmsp)/2
data = data.dropna(thresh = half_count, axis=1)
data = data.drop_duplicates()  
data = data.drop([680,691,692,693,694,695,696])


data['province'] = data.item_loc.apply(lambda x: x.split()[0])
data['city'] = data.item_loc.apply(lambda x: x.split()[0] \
                                if len(x) < 4 else x.split()[1])

data['sales'] = data.view_sales.apply(lambda x: x.split('人')[0])  
data['sales'] = data.sales.apply(lambda x: x.split('+')[0])
# 查看各列数据类型
data.dtypes   

# 将数据类型进行转换
data.to_csv("C:\\Users\\13407\\Desktop\\基金管理案例大赛\\taobao1.csv", encoding = "gb18030")
data = pd.read_csv("C:\\Users\\13407\\Desktop\\基金管理案例大赛\\taobao1.csv", encoding = "gb18030")
data = data.drop(columns=['Unnamed: 0'])
data['sales'] = data.sales.astype('int')                                                     

list_col = ['province','city']
for i in  list_col:
    data[i] = data[i].astype('category') 

# 删除不用的列：
data['price'] = data.view_price.apply(lambda x: x.split('元')[0])
data['price'] = data.price.astype('float')
data = data.drop(['item_loc','view_sales','view_price'], axis=1) 

#------------------------------------------------------------------------------------------------------------------------------------
##3.数据挖掘
title = data.raw_title.values.tolist()    #转为list

import jieba
title_s = []
for line in title:     
   title_cut = jieba.lcut(line)    
   title_s.append(title_cut)
stopwords = pd.read_excel('C:\\Users\\13407\\Desktop\\基金管理案例大赛\\原代码附件\\stopwords.xlsx')        
stopwords = stopwords.stopword.values.tolist()      

# 剔除停用词：
title_clean = []
for line in title_s:
   line_clean = []
   for word in line:
      if word not in stopwords:
         line_clean.append(word)
   title_clean.append(line_clean)

title_clean_dist = []  
for line in title_clean:   
   line_dist = []
   for word in line:
      if word not in line_dist:
         line_dist.append(word)
   title_clean_dist.append(line_dist)
 
allwords_clean_dist = []
for line in title_clean_dist:
   for word in line:
      allwords_clean_dist.append(word)

# 把列表 allwords_clean_dist 转为数据框： 
df_allwords_clean_dist = pd.DataFrame({'allwords': allwords_clean_dist})


# 对过滤_去重的词语 进行分类汇总：
word_count = df_allwords_clean_dist.allwords.value_counts().reset_index()    
word_count.columns = ['word','count']      #添加列名 

add_words = pd.read_excel('C:\\Users\\13407\\Desktop\\基金管理案例大赛\\原代码附件\\add_words.xlsx')     #导入整理好的待添加词语

# 添加词语： 
for w in add_words.word:
   jieba.add_word(w , freq=1000)  
import numpy as np   

w_s_sum = []
for w in word_count.word:
   i = 0
   s_list = []
   for t in title_clean_dist:
      if w in t:
         s_list.append(data.sales[i])
      i+=1
   w_s_sum.append(sum(s_list))     #list求和
   
df_w_s_sum = pd.DataFrame({'w_s_sum': w_s_sum})  

# 把 word_count 与对应的 df_w_s_sum 合并为一个表：
df_word_sum = pd.concat([word_count,df_w_s_sum], axis=1,ignore_index = True)
df_word_sum.columns = ['word','count','w_s_sum']     #添加列名 

df_word_sum.sort_values('w_s_sum',inplace=True,ascending=True)  #升序 
df_w_s = df_word_sum.tail(30)     #取最大的30行数据

import matplotlib
from matplotlib import pyplot as plt

font = {'family' : 'SimHei'}    #设置字体
matplotlib.rc('font', **font)

index = np.arange(df_w_s.word.size)
plt.figure(figsize=(6,12))
plt.barh(index, df_w_s.w_s_sum, color='purple', align='center', alpha=0.8) 
plt.yticks(index, df_w_s.word, fontsize=11)                             

# 添加数据标签：
for y,x in zip(index , df_w_s.w_s_sum):
   plt.text(x, y, '%.0f' %x , ha='left', va= 'center', fontsize=6)    
plt.show()

data_p = data[data['price'] < 20000]    

plt.figure(figsize=(7,5))
plt.hist(data_p['price'] ,bins=15 ,color='purple')   #分为15组  
plt.xlabel('价格',fontsize=12)
plt.ylabel('商品数量',fontsize=12)         
plt.title('不同价格对应的商品数量分布',fontsize=15)  
plt.show()  

data_s = data[data['sales'] > 100]    
print('销量100以上的商品占比: %.3f' %(len(data_s)/len(data)))
plt.figure(figsize=(7,5))
plt.hist(data_s['sales'] ,bins=20 , color='purple')    #分为20组  
plt.xlabel('销量', fontsize=12)
plt.ylabel('商品数量', fontsize=12)         
plt.title('不同销量对应的商品数量分布', fontsize=15)
plt.show()


data['price'] = data.view_price.astype('float')   
data['group'] = pd.qcut(data.price, 12)         
df_group = data.group.value_counts().reset_index()
df_s_g = data[['sales','group']].groupby('group').mean().reset_index()  
index = np.arange(df_s_g.group.size)
plt.figure(figsize=(4,4))
plt.bar(index, df_s_g.sales, color='purple')     
plt.xticks(index, df_s_g.group, fontsize=7, rotation=25) 
plt.xlabel('Group')
plt.ylabel('mean_sales')
plt.title('不同价格区间的商品的平均销量')
plt.show()


data_p = data[data['price'] < 700]  
fig, ax = plt.subplots(figsize=(8,5))    
ax.scatter(data_p['price'], data_p['sales'],color='purple')
ax.set_xlabel('价格')
ax.set_ylabel('销量')
ax.set_title('商品价格对销量的影响',fontsize=14)
plt.show()


data['GMV'] = data['price'] * data['sales']
import seaborn as sns
sns.regplot(x="price",y='GMV',data=data,color='purple') 
plt.figure(figsize=(8,4))
data.province.value_counts().plot(kind='bar',color='purple')
plt.xticks(rotation= 0, fontsize=7)       
plt.xlabel('省份')
plt.ylabel('数量')
plt.title('不同省份的商品店家分布')
plt.show()













