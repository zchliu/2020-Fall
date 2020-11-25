import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":

    word = input("输入关键词搜索：")
    news_url = 'https://s.weibo.com/weibo?q=' + word + '&xsort=hot&suball=1&Refer=SWeibo_box'
    r = requests.get(news_url)
    soup = BeautifulSoup(r.text, 'lxml')

    info = soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(1) > div > div.card-feed > div.content > p.txt')

    N = 10
    for i in range(N):
        try:
            info = soup.select('#pl_feedlist_index > div:nth-child(2) > div:nth-child(' + str(i + 1) + ') > div > div.card-feed > div.content > p.txt')
            print(info[0].get_text())
        except:
            pass


        