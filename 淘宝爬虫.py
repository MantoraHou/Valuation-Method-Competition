 # -*- coding: utf-8 -*-
import requests
import re
import pandas as pd
import time

# 此处写入登录之后自己的cookies
cookie = 't=0bbc032e24c571bf22d98df4213df36f; cna=PfHqFdpMmXwCAd9oCvkBFtte; isg=BIaGbCCDGK7T88_Id2Oq9paQ1HwI58qhLCtQhnCvcKmEcyaN2HQQsWgBTy-_QMK5; tfstk=cRiFBgXOlHKFFFqYxkZyOqUqnBrdZ_NuGGy4-2p6BftcEJaGiIf8IM8gQ87J2yf..; l=eBapSp57O7rCXozDBOfalurza779IIOYYuPzaNbMiOCP_7Cp5gxGW60nD3T9Cn1Vh6pyR3RBOI6JBeYBqIv4n5U62j-la_Hmn; sgcookie=E100PUgiMNUuz1pWsx55UPAXu2vnBQEGIIaEIqlmqtCtMwvhRXTbNCqZHgB%2BV9T0UKlpXdd%2BVyLI%2BLvu9HLcBSqYvFWmxo4HvlMeX1g3XJ8uL7vFFnztkI9D0XV96N9DG7H1; uc3=vt3=F8dCvCoi76YHRsWpIjo%3D&nk2=CyKrc74DoesxO0YQGr…e; _samesite_flag_=true; csg=58504149; cancelledSubSites=empty; dnk=houshiping19961019; skt=e5e73b9847f9ea54; existShop=MTY0NzYxNjkzNA%3D%3D; v=0; JSESSIONID=4967D3471ACB3EA3B123A7FEC233E72C; alitrackid=www.taobao.com; lastalitrackid=www.taobao.com; _m_h5_tk=361278ef5ceb56723d179c34f75f4316_1648311372852; _m_h5_tk_enc=6a8083d8d34c229e95ef09d42882cbf0; uc1=pas=0&existShop=false&cookie21=WqG3DMC9Edo1SB5NB6Qtng%3D%3D&cookie16=V32FPkk%2FxXMk5UvIbNtImtMfJQ%3D%3D&cookie14=UoewCLKrNjgjgQ%3D%3D; mt=ci=-1_0; thw=cn‘
# 获取页面信息
def getHTMLText(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0'}
    user_cookies = cookie
    cookies = {}
    for a in user_cookies.split(';'):  # 因为cookies是字典形式，所以用spilt函数将之改为字典形式
        name, value = a.strip().split('=', 1)
        cookies[name] = value
    try:
        r = requests.get(url, cookies=cookies, headers=headers, timeout=60)
        print(r.status_code)
        print(r.cookies)
        return r.text
    except:
        print('获取页面信息失败')
        return ''

#  格式化页面，查找数据
def parsePage(html):
    list = []
    try:
        views_title = re.findall('"raw_title":"(.*?)","pic_url"', html)
        print(len(views_title))  # 打印检索到数据信息的个数，如果此个数与后面的不一致，则数据信息不能加入列表
        print(views_title)
        views_price = re.findall('"view_price":"(.*?)","view_fee"', html)
        print(len(views_price))
        print(views_price)
        item_loc = re.findall('"item_loc":"(.*?)","view_sales"', html)
        print(len(item_loc))
        print(item_loc)
        views_sales = re.findall('"view_sales":"(.*?)","comment_count"', html)
        print(len(views_sales))
        print(views_sales)
        comment_count = re.findall('"comment_count":"(.*?)","user_id"', html)
        print(len(comment_count))
        print(comment_count)
        shop_name = re.findall('"nick":"(.*?)","shopcard"', html)
        print(len(shop_name))
        for i in range(len(views_price)):
            list.append([views_title[i], views_price[i], item_loc[i], comment_count[i], views_sales[i], shop_name[i]])
        # print(list)
        print('爬取数据成功')
        return list
    except:
        print('有数据信息不全，如某一页面中某一商品缺少地区信息')

# 存储到csv文件中，为接下来的数据分析做准备
def save_to_file(list):
    data = pd.DataFrame(list)
    data.to_csv('C:\\Users\\13407\\Desktop\\基金管理案例大赛\\taobao_jiweiiu.csv', header=False, mode='a+')  # 用追加写入的方式

def main():
    name = [['views_title', 'views_price', 'item_loc', 'comment_count', 'views_sales', 'shop_name']]
    data_name = pd.DataFrame(name)
    data_name.to_csv('C:\\Users\\13407\\Desktop\\基金管理案例大赛\\taobao_jiweiiu.csv', header=False, mode='a+')  # 提前保存一行列名称
    goods = input('请输入想查询的商品名称：'.strip())  # 输入想搜索的商品名称
    depth = 5  # 爬取的页数
    start_url = 'http://s.taobao.com/search?q=' + goods  # 初始搜索地址
    for i in range(depth):
        time.sleep(3 + i)
        try:
            page = i + 1
            print('正在爬取第%s页数据' % page)
            url = start_url + 'imgfile=&js=1&stats_click=search_radio_all%3A1&initiative_id=staobaoz_20200408&ie=utf8&sort=sale-desc&bcoffset=0&p4ppushleft=%2C44&s=' + str(44 * i)
            html = getHTMLText(url)
            # print(html)
            list = parsePage(html)
            save_to_file(list)
        except:
            print('数据没保存成功')

if __name__ == '__main__':
    main()
#https://s.taobao.com/search?initiative_id=tbindexz_20170306&ie=utf8&spm=a21bo.jianhua.201856-taobao-item.2&sourceId=tb.index&search_type=item&ssid=s5-e&commend=all&imgfile=&q=%E9%A2%84%E8%B0%83%E9%B8%A1%E5%B0%BE%E9%85%92&suggest=history_1&_input_charset=utf-8&wq=%E9%A2%84%E8%B0%83&suggest_query=%E9%A2%84%E8%B0%83&source=suggest
