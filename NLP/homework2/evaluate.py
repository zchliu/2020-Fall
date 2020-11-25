import text_classify
import preprocessing
import numpy as np
import os
import time


class item:
    def __init__(self, fk=np.zeros(9, dtype=int), nk=0, tfidf=np.zeros(9)):
        self.fk = fk.copy()
        self.nk = nk
        self.tfidf = tfidf.copy()


def evaluate(test_dir):

    clas_list = os.listdir(test_dir)
    clas_num = len(clas_list)

    total_predict_correct = 0
    total_doc_num = 0
    for i in range(clas_num):

        test_clas_dir = test_dir + "//" + clas_list[i] + "//"
        doc_list = os.listdir(test_clas_dir)
        doc_num = len(doc_list)
        total_doc_num += doc_num
        predict_correct = 0

        begin_time = time.time()
        for j in range(doc_num):

            # content = preprocessing.readfile(test_clas_dir + doc_list[j])
            # content = re.sub(preprocessing.pattern, ' ', content)
            # content_seg = jieba.cut(content)

            predicted_clas = text_classify.predict(test_clas_dir + doc_list[j])
            if (clas_list[i] == predicted_clas):
                predict_correct += 1
                total_predict_correct += 1

        end_time = time.time()
        correct_rate = predict_correct / doc_num
        # print(str(clas_list[i]) + "'s correct rate is " + str(correct_rate))
        print("{0}'s correct rate is {1:.2%}.consuming time：{2:.2f}".format(str(clas_list[i]), correct_rate, end_time - begin_time))

    total_correct_rate = total_predict_correct / total_doc_num
    print("total correct rate is {0:.2%}".format(total_correct_rate))


if __name__ == "__main__":
    evaluate(r"D:\2020-fall\NLP\homework2\test")