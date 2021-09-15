# -*- coding:utf-8 -*- 

import pickle

with open("filenames.pickle", 'rb') as f:
    class_id = pickle.load(f)
    print(class_id[0:3])