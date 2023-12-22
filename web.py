import requests
import pandas as pd
import json
from threading import Thread

test_data = pd.read_csv('./data/test.feature.csv')
train_data = pd.read_csv('./data/train.news.csv')
print("finish read csv")
def is_wrong(url):
    re=requests.get(url=url).text
    return '该内容已被发布者删除' in re
def save_html(url,id,data:str):
    re=requests.get(url=url).text
    f=open('data/html/'+data+'/'+str(id)+'.html','w',encoding='utf-8')
    f.write(re)
    f.close()
def check_urls(urls, result):
    index=0
    length=len(urls)
    for url in urls:
        index+=1
        if index%(length//100)==0 :
            print(index//(length//100),"%")
        result[index]=int(is_wrong(url=url))
    print("done!")
def save_urls(urls,data:str):
    index=0
    length=len(urls)
    for url in urls:
        index+=1
        if index%(length//100)==0 :
            print(index//(length//100),"%")
        save_html(url,index,data)
    print("done!")

threads = []
test_url={}
train_url={}
threads.append(Thread(target=check_urls, args=(train_data['News Url'], train_url)))
threads.append(Thread(target=check_urls, args=(test_data['News Url'], test_url)))
threads.append(Thread(target=save_urls, args=(train_data['News Url'], 'train')))
threads.append(Thread(target=save_urls, args=(test_data['News Url'], 'test')))
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

train_text=json.dumps(train_url)
test_text=json.dumps(test_url)
with open('./data/url_check.txt','w') as f:
    f.write(train_text)
    f.write('\n')
    f.write(test_text)