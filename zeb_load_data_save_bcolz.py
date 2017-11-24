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


def load_one_classes_data(filepath, times):
    
    files = [f for f in os.listdir(filepath)]
    num_data = len(files)
    example_spec = make_spec_normalized(filepath+files[0])
    spec_shape = example_spec.shape

    buffer = np.zeros((num_data, spec_shape[0], spec_shape[1]), dtype=np.float32)
    
    for i, file in enumerate(files):
        spec = make_spec_normalized(filepath + file)
        buffer[i,:,:] = spec
        
    return np.repeat(buffer, times, axis=0)

def load_all_data(filepath):
    dirs = ['left', 'up', 'down', 'no', 'right', 'on', 'yes', 'off', 'unknown', 'silence', 'stop', 'go']
    times = {'left':2, 'up':2, 'down':2,
             'no':2, 'right':2, 'on':2,
             'yes':2, 'off':2, 'unknown':1,
             'silence':20, 'stop':2, 'go':2}
    
    data_list = []
    class_list = []
    
    for cls, d in enumerate(dirs):
        print('loading', d)
        data = load_one_classes_data(filepath + d + '/', times[d])
        num_data = data.shape[0]
        classes = np.ones((num_data,1), dtype=np.float32) * cls
        data_list.append(data)
        class_list.append(classes)
        
    return np.vstack(data_list), np.vstack(class_list)


TRAIN_PATH = 'train/train_train/'
VALID_PATH = 'train/train_valid/'   
TEST_PATH = 'train/train_valid/'  
OUTPUT_PATH = 'train/bcolz/'


ALL_PATHS = [TRAIN_PATH, VALID_PATH, TEST_PATH]
ALL_NAMES = 'train/ valid/ test/'.split()


for path, name in zip(ALL_PATHS, ALL_NAMES):
    X, Y = load_all_data(path) 
    x_disk = bcolz.carray(X, dtype='float32', rootdir=OUTPUT_PATH + name + 'data', expectedlen=sys.getsizeof(X))
    y_disk = bcolz.carray(Y, dtype='float32', rootdir=OUTPUT_PATH + name + 'labels', expectedlen=sys.getsizeof(Y))
    x_disk.flush()
    y_disk.flush()
    del X, Y, x_disk, y_disk
    print('Finished saving', name)
    




    
