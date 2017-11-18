# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:51:21 2017

@author: Edward
"""

import os
import shutil

def prepare_data(directory):
    valid_sub_directories = 'yes no up down left right on off stop go silence unknown'.split()
    for sub_directory in os.listdir(directory):
        if sub_directory in valid_sub_directories:
            continue
        else:
            for file in os.listdir(directory + sub_directory):
                shutil.move(directory + sub_directory + '/' + file,
                            directory + 'unknown/' + sub_directory + '_' + file)
        
        os.removedirs(directory + sub_directory)
        
TRAIN_PATH = 'train/train_train/'
VALID_PATH = 'train/train_valid/'
TEST_PATH = 'train/train_test/'        
paths = [TRAIN_PATH, VALID_PATH, TEST_PATH]        
        
for path in paths:    
    prepare_data(path)    