import os
from collections import defaultdict
import pickle
import numpy as np
import preprocessing


class item:
    def __init__(self, fk=np.zeros(9, dtype=int), nk=0, tfidf=np.zeros(9)):
        self.fk = fk.copy()
        self.nk = nk
        self.tfidf = tfidf.copy()

def predict(path):
    if (not os.path.exists("dictionary.pickle")):
        print("dictionaty.pickle doesn't exist!please first build dictionary!")
    else:
        dictionary_file = open("dictionary.pickle", "rb")
        dictionary, N_list, clas_list, frequency_list = pickle.load(dictionary_file)
        dictionary_file.close()

        clas_len = len(clas_list)
        input_path = path
        words_dict = preprocessing.build_eigen(input_path)
        # print(words_dict)

        predicted_clas = ""
        Pmax = -100000000000
        for i in range(clas_len):
            P = 0
            P += np.log(N_list[i] / np.sum(N_list))
            for word in words_dict:
                # print(words_dict[word] * np.log((dictionary[word].tfidf[i] + 1e-6) / (weight_sum + 1e-6)))
                P += words_dict[word] * np.log(dictionary[word].tfidf[i] + 1e-10)

            # print(i, P)
            if (P > Pmax):
                predicted_clas = clas_list[i]
                Pmax = P

        return predicted_clas

def view_dictionary():
    if (not os.path.exists("dictionary.pickle")):
        print("dictionaty.pickle doesn't exist!please first build dictionary!")
    else:
        dictionary_file = open("dictionary.pickle", "rb")
        dictionary, N_list, clas_list, frequency_list = pickle.load(dictionary_file)
        dictionary_file.close()

        i = 8
        print("|" + clas_list[i] + "|" + "tfidf" + "|")
        print("|---|---|")
        dic_sorted = sorted(dictionary.items(), key=lambda item: item[1].tfidf[i], reverse=True)
        k = 0
        for word, word_info in dic_sorted:
            print("|" + word + "|" + str(word_info.tfidf[i]) + "|")
            k += 1
            if k > 20:
                break

if __name__ == "__main__":

    view_dictionary()
    # input_path = input("please enter the filepath.")
    # predicted_clas = predict(input_path)
    # print(predicted_clas)
