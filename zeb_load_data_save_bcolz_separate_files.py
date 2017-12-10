# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:53:52 2017

@author: Edward
"""

import os
import sys
import numpy as np
import bcolz
from data_utils import make_spec_normalized


def load_one_classes_data(filepath):
    
    files = [f for f in os.listdir(filepath)]
    num_data = len(files)
    example_spec = make_spec_normalized(filepath+files[0])
    spec_shape = example_spec.shape

    buffer = np.zeros((num_data, spec_shape[0], spec_shape[1]), dtype=np.float32)
    
    for i, file in enumerate(files):
        spec = make_spec_normalized(filepath + file)
        buffer[i,:,:] = spec
        
    return buffer


TRAIN_PATH = 'train/train_train/'
VALID_PATH = 'train/train_valid/'   
TEST_PATH = 'train/train_valid/'  
OUTPUT_PATH = 'train/bcolz_separate/'


ALL_PATHS = [TRAIN_PATH, VALID_PATH, TEST_PATH]
ALL_NAMES = 'train/ valid/ test/'.split()

for path, name in zip(ALL_PATHS, ALL_NAMES):
    directories = ['left', 'up', 'down', 'no', 'right', 'on', 'yes', 'off', 'unknown', 'silence', 'stop', 'go']
    for directory in directories:        
        X = load_one_classes_data(path + directory + '/') 
        os.mkdir(OUTPUT_PATH + name + directory)
        x_disk = bcolz.carray(X, dtype='float32', rootdir=OUTPUT_PATH + name + directory + '/data', expectedlen=sys.getsizeof(X))
        x_disk.flush()
        del X, x_disk
        print('Finished saving', name, directory)
    




    
