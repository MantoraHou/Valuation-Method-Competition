# -*- coding: utf-8 -*-
from pyquery import PyQuery as pq
import time
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

def get_text(page_source):
    html = page_source
    doc = pq(html)
    items = doc('#mainsrp-itemlist .items .item').items()
    for item in items:
        product = {
            'image': item.find('.pic .img').attr('data-src'),
            'price': item.find('.price').text(),
            'deal': item.find('.deal-cnt').text(),
            'title': item.find('.title').text(),
            'shop': item.find('.shop').text(),
            'location': item.find('.location').text()
            }
        yield product.values()
        
if __name__=='__main__':
    file=open("C:\\Users\\13407\\Desktop\\基金管理案例大赛\\taobao_jiweiiu.csv","w",encoding='utf-8',newline="")
    f=csv.writer(file)
    f.writerow(['image','price','deal','title','shop','location'])
    url = 'https://s.taobao.com/search'
    brown=webdriver.Chrome()
    brown.get(url)
    time.sleep(10)
    for i in range(0,100):
    	try:
            url="https://s.taobao.com/search?ie=utf8&initiative_id=staobaoz_20190920&stats_click=search_radio_all%3A1&js=1&imgfile=&q=%E9%9B%B6%E9%A3%9F&suggest=0_1&_input_charset=utf-8&wq=LINGSHI&suggest_query=LINGSHI&source=suggest&bcoffset=3&ntoffset=3&p4ppushleft=1%2C48&s="+str(i*44)
            brown.get(url)
            for t in get_text(brown.page_source):
                f.writerow(t)
        except:
        	print('爬取完成')
        time.sleep(1)
    file.close()
    
