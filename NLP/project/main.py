from crawler import UrlDownLoader
from crawler import HtmlOuter
from crawler import HtmlParser
from crawler import UrlManager

import urllib
import os
import pickle
import time
import random
import json


class MainCrawler():
    def __init__(self):
        # 初始值，实例化四大处理器：url管理器，下载器，解析器，输出器
        self.urls = UrlManager()
        self.downloader = UrlDownLoader()
        self.parser = HtmlParser()
        self.outer = HtmlOuter()
        # 开始爬虫方法

    def start_craw(self, main_url):
        print('爬虫开始...')
        count = 1
        self.urls.add_new_url(main_url)
        while self.urls.has_new_url():
            try:
                time.sleep(0.1)
                new_url = self.urls.get_new_url()
                print('爬虫%d,%s' % (count, new_url))
                html_cont = self.downloader.down_load(new_url)
                new_urls, new_data = self.parser.parse(new_url, html_cont)
                # 将解析出的url放入url管理器，解析出的数据放入输出器中
                self.urls.add_new_urls(new_urls)
                self.outer.conllect_data(new_data)
                if count >= 10000:  # 控制爬取的数量
                    break
                count += 1
            except:
                print('爬虫失败一条')
        # self.outer.output_commandline()
        self.outer.create_dictionary()
        print('爬虫结束。')


def get_url(word):

    return 'https://baike.baidu.com/item/' + urllib.parse.quote(word)


def view_dictionary():
    if (not os.path.exists("dictionary.pickle")):
        print("dictionaty.pickle doesn't exist!please first build dictionary!")
    else:
        dictionary_file = open("dictionary.pickle", "rb")
        dictionary = pickle.load(dictionary_file)
        dictionary_file.close()

        word = random.choice(list(dictionary))
        print(word)
        print(dictionary[word])
        # for word, word_info in dictionary.items():

        #     print(word + ": " + word_info)


def save_as_json():
    if (not os.path.exists("dictionary.pickle")):
        print("dictionaty.pickle doesn't exist!please first build dictionary!")
    else:
        dictionary_file = open("dictionary.pickle", "rb")
        dictionary = pickle.load(dictionary_file)
        dictionary_file.close()

    dictionary = json.dumps(dictionary, ensure_ascii=False)
    with open("dictionary.json", mode='w', encoding='utf-8') as file_obj:
        file_obj.write(dictionary)


def read_json():
    if (not os.path.exists("dictionary.json")):
        print("dictionaty.json doesn't exist!please first build dictionary!")

    with open('dictionary.json') as file_obj:
        dictionary = json.load(file_obj)

    word = random.choice(list(dictionary))
    print(word)
    print(dictionary[word])


if __name__ == '__main__':

    # main_url = 'https://baike.baidu.com/item/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/217599'
    # main_url = 'https://baike.baidu.com/item/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/9180'
    # word = input("输入关键词：")
    # mc = MainCrawler()
    # mc.start_craw(get_url(word))

    # main_url = 'https://baike.baidu.com/item/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86'
    # mc = MainCrawler()
    # mc.start_craw(main_url)
    # view_dictionary()
    # save_as_json()
    read_json()
