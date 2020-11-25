#pl_top_realtimehot > table > tbody > tr:nth-child(1) > td.td-02

# https://s.weibo.com/top/summary/
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    news = []
    # 新建数组存放热搜榜
    hot_url = 'https://s.weibo.com/top/summary/'
    # 热搜榜链接
    r = requests.get(hot_url)
    # 向链接发送get请求获得页面
    soup = BeautifulSoup(r.text, 'lxml')
    # 解析页面

    urls_titles = soup.select('#pl_top_realtimehot > table > tbody > tr > td.td-02 > a')
    hotness = soup.select('#pl_top_realtimehot > table > tbody > tr > td.td-02 > span')
    print(urls_titles)
    for i in range(len(urls_titles) - 1):
        hot_news = {}
        # 将信息保存到字典中
        hot_news['title'] = urls_titles[i + 1].get_text()
        # get_text()获得a标签的文本
        hot_news['url'] = "https://s.weibo.com" + urls_titles[i]['href']
        # ['href']获得a标签的链接，并补全前缀
        hot_news['hotness'] = hotness[i].get_text()
        # 获得热度文本
        news.append(hot_news)
        # 字典追加到数组中

    for i in range(len(news)):

        print(news[i]['title'] + ": " + news[i]['hotness'])
