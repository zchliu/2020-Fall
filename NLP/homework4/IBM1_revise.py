# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 22:05:58 2020

@author: cjh
"""
from collections import Counter
import re
from tkinter import _flatten
import pickle


def preprocess(en_path, f_path):
    en_file = open(en_path, encoding='utf-8', errors='ignore')
    f_file = open(f_path, encoding='utf-8', errors='ignore')  

    en_list = []
    f_list = []

    for line in en_file:
        en_list.append(line)

    for line in f_file:
        f_list.append(line)

    en_rule = re.compile(u'[^a-zA-Z\s]')
    f_rule = re.compile(u'[^\u4e00-\u9fa5\s]')

    for i in range(len(en_list)):
        en_list[i] = en_rule.sub('', en_list[i])
        en_list[i] = en_list[i].split()
        en_list[i].insert(0, "_")

    for i in range(len(f_list)):
        f_list[i] = f_rule.sub('', f_list[i])
        f_list[i] = f_list[i].split()

    return en_list, f_list


en, cn = preprocess('en.txt', 'cn.txt')


def alignment(size, trials):
    S = cn[0:size]
    T = en[0:size]
    p_s_t = {}
    c_s_t = {}
    total_T = {}
    total_S = {}

    S_flatten = _flatten(S)
    T_flatten = _flatten(T)

    counter_S = Counter(S_flatten)
    counter_T = Counter(T_flatten)

    # initial
    for word_S in counter_S.keys():
        p_s_t[word_S] = {}
        c_s_t[word_S] = {}
        total_S[word_S] = 0

    for i in range(size):
        for word_S in S[i]:
            for word_T in T[i]:
                p_s_t[word_S].setdefault(word_T, 1)
                c_s_t[word_S].setdefault(word_T, 0)

    for word_T in counter_T.keys():
        total_T[word_T] = 0




    for trial in range(trials):
        print(trial)
        c_s_t_dist = c_s_t
        total_T_dist = total_T
        total_S_dist = total_S

        for z in range(size):
            for word_S in S[z]:
                total_S_dist[word_S] = 0
                for word_T in T[z]:
                    total_S_dist[word_S] = total_S_dist[word_S] + p_s_t[word_S][word_T]
            for word_S in S[z]:
                for word_T in T[z]:
                    c_s_t_dist[word_S][word_T] = c_s_t_dist[word_S][word_T] + p_s_t[word_S][word_T] / total_S_dist[
                        word_S]
                    total_T_dist[word_T] = total_T_dist[word_T] + p_s_t[word_S][word_T] / total_S_dist[word_S]
        
        for word_S in counter_S.keys():
            for word_T in p_s_t[word_S].keys():
                p_s_t[word_S][word_T] = c_s_t_dist[word_S][word_T] / total_T_dist[word_T]
        
          
    return p_s_t


result = alignment(100000, 50)
print("finish")
result_revise_1_file = open('result_1.pickle', 'wb')
pickle.dump(result, result_revise_1_file)
result_revise_1_file.close()

test_S=cn[99995]
test_T=en[99995]

for word_S in test_S:
    print(word_S,max(result[word_S],key=result[word_S].get))




