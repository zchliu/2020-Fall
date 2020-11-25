import shutil
import os
import random

src_dir = r"D:\Fudan_dataset"
dst_dir1 = r"D:\2020-fall\NLP\homework2\train"
dst_dir2 = r"D:\2020-fall\NLP\homework2\test"

if (not os.path.exists(dst_dir1)):
    os.mkdir(dst_dir1)
if (not os.path.exists(dst_dir2)):
    os.mkdir(dst_dir2)

clas_list = os.listdir(src_dir)
for i in range(len(clas_list)):
    if (not os.path.exists(dst_dir1 + "\\" + clas_list[i])):
        os.mkdir(dst_dir1 + "\\" + clas_list[i])
    if (not os.path.exists(dst_dir2 + "\\" + clas_list[i])):
        os.mkdir(dst_dir2 + "\\" + clas_list[i])

    doc_list = os.listdir(src_dir + "\\" + clas_list[i])
    for j in range(len(doc_list)):
        if random.uniform(0, 1) < 0.8:
            shutil.copyfile(src_dir + "\\" + clas_list[i] + "\\" + doc_list[j], dst_dir1 + "\\" + clas_list[i] + "\\" + doc_list[j])
        else:
            shutil.copyfile(src_dir + "\\" + clas_list[i] + "\\" + doc_list[j], dst_dir2 + "\\" + clas_list[i] + "\\" + doc_list[j])

    # clas_list = os.listdir(data_dir)
#  shutil.copyfile(r"D:\Fudan_dataset\C3-Art\C3-Art0001.txt", r"D:\2020-fall\NLP\homework2\train\C3-Art0001.txt")
