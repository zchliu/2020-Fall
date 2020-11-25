import numpy as np
from collections import defaultdict
import re
import pickle
import os
import time

cn_pattern = re.compile(r'[^\u4e00-\u9fa5]')
en_pattern = re.compile(r'[^a-zA-Z\s]')


def preprocess(en_path, cn_path):

    with open(en_path, "rb") as ef:

        en_list = []
        row = ef.readline().decode('utf-8')
        while row != "":
            row = re.sub(en_pattern, ' ', row)
            en_list.append(row)
            row = ef.readline().decode('utf-8')

    with open(cn_path, "rb") as cf:

        cn_list = []
        row = cf.readline().decode('utf-8')
        while row != "":
            row = re.sub(cn_pattern, ' ', row)
            cn_list.append(row)
            row = cf.readline().decode('utf-8')

    return en_list, cn_list


def build_dictionary1(en_path, cn_path):

    begin_time = time.time()

    en_list, cn_list = preprocess(en_path, cn_path)
    length = len(en_list)

    dictionary_t = defaultdict(float)
    iteration = 20

    while (iteration > 0):

        iteration -= 1
        print(20 - iteration)

        dictionary_count = defaultdict(float)
        dictionary_total = defaultdict(float)

        for i in range(length):

            en_sentence = en_list[i]
            cn_sentence = cn_list[i]
            en_words = en_sentence.split()
            cn_words = cn_sentence.split()
            cn_words.append("_")

            words_total_list = [0 for m in range(len(en_words))]
            for j in range(len(en_words)):
                words_total = 0
                for k in range(len(cn_words)):

                    dictionary_t.setdefault((cn_words[k], en_words[j]), 1)
                    words_total += dictionary_t[(cn_words[k], en_words[j])]

                words_total_list[j] = words_total

            for j in range(len(en_words)):
                for k in range(len(cn_words)):
                    dictionary_count[(cn_words[k], en_words[j])] += dictionary_t[(cn_words[k], en_words[j])] / words_total_list[j]
                    dictionary_total[cn_words[k]] += dictionary_t[(cn_words[k], en_words[j])] / words_total_list[j]

        for (cn_word, en_word) in dictionary_count.keys():
            dictionary_t[(cn_word, en_word)] = dictionary_count[(cn_word, en_word)] / dictionary_total[cn_word]

    dictionary_file = open("dictionary1.pickle", "wb")
    pickle.dump(dictionary_t, dictionary_file)
    dictionary_file.close()

    end_time = time.time()
    print("The dictionary1 has been built successfully! Total time consuming: " + str(end_time - begin_time))


def alignment1(en_str, cn_str):

    # 求解从中文到英文的最优对齐，输入为中英文的分词形式，无标点，输出为每一个英文单词对应的对齐中文单词

    if (not os.path.exists("dictionary1.pickle")):
        print("dictionaty1.pickle doesn't exist!please first build dictionary!")
    else:
        dictionary_file = open("dictionary1.pickle", "rb")
        dictionary = pickle.load(dictionary_file)
        dictionary_file.close()

        en_str = re.sub(en_pattern, ' ', en_str)
        cn_str = re.sub(cn_pattern, ' ', cn_str)
        en_list = en_str.split()
        cn_list = cn_str.split()
        cn_list.append("_")

        n = len(cn_list)
        m = len(en_list)

        a = [0 for i in range(m)]
        for i in range(m):
            prob_max = -1
            a[i] = -1
            for j in range(n):
                prob = dictionary[(cn_list[j], en_list[i])]
                if prob > prob_max:
                    prob_max = prob
                    a[i] = j

        for i in range(m):
            print(en_list[i] + ": " + cn_list[a[i]])


def view_dictionary1():
    if (not os.path.exists("dictionary.pickle")):
        print("dictionaty.pickle doesn't exist!please first build dictionary!")
    else:
        dictionary_file = open("dictionary.pickle", "rb")
        dictionary = pickle.load(dictionary_file)
        dictionary_file.close()

        dic_sorted = sorted(dictionary.items(), key=lambda item: item[1], reverse=True)
        k = 0
        for word, word_info in dic_sorted:
            print("|" + str(word) + "|" + str(word_info) + "|")
            k += 1
            if k > 50:
                break


def build_dictionary2(en_path, cn_path):

    begin_time = time.time()

    en_list, cn_list = preprocess(en_path, cn_path)
    length = len(en_list)

    dictionary_t = defaultdict(float)
    dictionary_q = defaultdict(float)
    iteration = 20

    while (iteration > 0):

        iteration -= 1
        print(20 - iteration)

        dictionary_count = defaultdict(float)
        dictionary_total = defaultdict(float)
        dictionary_count_num = defaultdict(float)
        dictionary_total_num = defaultdict(float)

        for i in range(length):

            en_sentence = en_list[i]
            cn_sentence = cn_list[i]
            en_words = en_sentence.split()
            cn_words = cn_sentence.split()
            cn_words.append("_")

            words_total_list = [0 for m in range(len(en_words))]
            for j in range(len(en_words)):
                words_total = 0
                for k in range(len(cn_words)):

                    dictionary_t.setdefault((cn_words[k], en_words[j]), 1)
                    dictionary_q.setdefault((k, j, len(cn_words), len(en_words)), 1)
                    words_total += dictionary_t[(cn_words[k], en_words[j])]

                words_total_list[j] = words_total

            flag = 0
            for j in range(len(en_words)):

                if flag == 0:
                    flag = 1
                    denominator = 0
                    for k in range(len(cn_words)):
                        denominator = denominator + dictionary_q[(k, j, len(cn_words), len(en_words))] * dictionary_t[(cn_words[k], en_words[j])]

                for k in range(len(cn_words)):

                    delta = dictionary_q[(k, j, len(cn_words), len(en_words))] * dictionary_t[(cn_words[k], en_words[j])] / denominator
                    dictionary_count[(cn_words[k], en_words[j])] += delta
                    dictionary_total[cn_words[k]] += delta
                    dictionary_count_num[(k, j, len(cn_words), len(en_words))] += delta
                    dictionary_total_num[(j, len(cn_words), len(en_words))] += delta

        for (cn_word, en_word) in dictionary_count.keys():
            dictionary_t[(cn_word, en_word)] = dictionary_count[(cn_word, en_word)] / dictionary_total[cn_word]

        for (k, j, cn_len, en_len) in dictionary_q.keys():
            dictionary_q[(k, j, cn_len, en_len)] = dictionary_count_num[(k, j, cn_len, en_len)] / dictionary_total_num[(j, cn_len, en_len)]

    dictionary_file = open("dictionary2_t.pickle", "wb")
    pickle.dump(dictionary_t, dictionary_file)
    dictionary_file.close()

    dictionary_file = open("dictionary2_q.pickle", "wb")
    pickle.dump(dictionary_t, dictionary_file)
    dictionary_file.close()

    end_time = time.time()
    print("The dictionary has been built successfully! Total time consuming: " + str(end_time - begin_time))


def view_dictionary2():
    if (not os.path.exists("dictionary2_t.pickle")):
        print("dictionaty2_t.pickle doesn't exist!please first build dictionary!")
    else:
        dictionary_file = open("dictionary2_t.pickle", "rb")
        dictionary_t = pickle.load(dictionary_file)
        dictionary_file.close()

        dic_sorted = sorted(dictionary_t.items(), key=lambda item: item[1], reverse=True)
        k = 0
        for word, word_info in dic_sorted:
            print("|" + str(word) + "|" + str(word_info) + "|")
            k += 1
            if k > 50:
                break


def alignment2(en_str, cn_str):

    # 求解从中文到英文的最优对齐，输入为中英文的分词形式，无标点，输出为每一个英文单词对应的对齐中文单词

    if (not os.path.exists("dictionary2_t.pickle")):
        print("dictionaty2_t.pickle doesn't exist!please first build dictionary!")
    else:
        dictionary_file = open("dictionary2_t.pickle", "rb")
        dictionary_t = pickle.load(dictionary_file)
        dictionary_file.close()

        dictionary_file = open("dictionary2_q.pickle", "rb")
        dictionary_q = pickle.load(dictionary_file)
        dictionary_file.close()

        en_str = re.sub(en_pattern, ' ', en_str)
        cn_str = re.sub(cn_pattern, ' ', cn_str)
        en_list = en_str.split()
        cn_list = cn_str.split()
        cn_list.append("_")

        n = len(cn_list)
        m = len(en_list)

        a = [0 for i in range(m)]
        for i in range(m):
            prob_max = -1
            a[i] = -1
            for j in range(n):
                prob = dictionary_t[(cn_list[j], en_list[i])] 
                if prob > prob_max:
                    prob_max = prob
                    a[i] = j

        for i in range(m):
            print(en_list[i] + ": " + cn_list[a[i]])


if __name__ == "__main__":

    # build_dictionary1(r'D:\2020-fall\NLP\homework4\en.txt', r'D:\2020-fall\NLP\homework4\cn.txt')
    # view_dictionary1()
    alignment1(" A science fiction cannot be regarded as a mere entertainment, but in fact it tells the reader much more." ," 科幻小说 不能 简单 地 看成是 供 消遣 的 ， 而 实际上 它 给 读者 展示 更 深刻 的 内容 。 " )
    print("----------")
    alignment1(" Do you eat", "你 吃 了 吗")
    print("----------")
    alignment1(" Boys, are your hands clean?", " 孩子 们 ， 你们 的 手 干净 吗 ？ ")
    print("----------")
    # build_dictionary2(r'D:\2020-fall\NLP\homework4\en.txt', r'D:\2020-fall\NLP\homework4\cn.txt')
    # view_dictionary2()
    alignment2(" A science fiction cannot be regarded as a mere entertainment, but in fact it tells the reader much more." , " 科幻小说 不能 简单 地 看成是 供 消遣 的 ， 而 实际上 它 给 读者 展示 更 深刻 的 内容 。 " )
    print("----------")
    alignment2(" Do you eat", "你 吃 了 吗")
    print("----------")
    alignment2(" Boys, are your hands clean?", " 孩子 们 ， 你们 的 手 干净 吗 ？ ")
    print("----------")
