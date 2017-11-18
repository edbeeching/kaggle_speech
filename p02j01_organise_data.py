# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 07:56:06 2017

@author: Edward
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt

CLASSES = 'yes no up down left right on off stop go silence unknown'.split()
CLASSES_SET = set(CLASSES)
TRAIN_BASE = 'train/audio/'
DIRECTORIES =  [d for d in os.listdir(TRAIN_BASE)]
VALID_BASE = 'train/train_valid/'
TEST_BASE = 'train/train_test/'
NEW_TRAIN_BASE = 'train/train_train/'
# get the testing and validation lists
# write to three directories

with open('train/validation_list.txt','r') as f:
    validation_files = f.readlines()
with open('train/testing_list.txt','r') as f:
    testing_files = f.readlines()
    

    
for directory in DIRECTORIES:
    os.makedirs(VALID_BASE + directory, exist_ok=True)
    os.makedirs(TEST_BASE + directory, exist_ok=True)

for file in validation_files:
    shutil.copyfile(TRAIN_BASE + file[:-1], VALID_BASE +  file[:-1])

for file in testing_files:
    shutil.copyfile(TRAIN_BASE + file[:-1], TEST_BASE +  file[:-1])
    
# Copy training data    
    
for file in validation_files:
    os.remove(NEW_TRAIN_BASE + file[:-1])

for file in testing_files:
    os.remove(NEW_TRAIN_BASE + file[:-1])
    
    
    