# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 07:26:02 2017

@author: Edward
"""

import os
import numpy as np
import matplotlib.pyplot as plt

CLASSES = 'yes no up down left right on off stop go silence unknown'.split()
CLASSES_SET = set(CLASSES)
TRAIN_BASE = 'train/audio/'
DIRECTORIES =  [d for d in os.listdir(TRAIN_BASE)]


class_dict = {}

for directory in DIRECTORIES:
    print(directory)
    files = [f for f in os.listdir(TRAIN_BASE+directory)]
    if directory in CLASSES_SET:
        label = directory
    else:
        label = 'unknown'
    if label in class_dict:
        class_dict[label] += len(files)
    else:
        class_dict[label] = len(files)

vals = [v for k,v in class_dict.items()]
keys = [i for i in range(len(class_dict))]
labels = [k for k,v in class_dict.items()]
        
fig, ax = plt.subplots()       
plt.bar(keys,vals)
ax.set_xticks(keys)
ax.set_xticklabels(labels)
