# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 10:27:55 2017

@author: Edward
"""

import os 
import numpy as np
import random
import bcolz
from scipy.io.wavfile import read as wavread
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from more_itertools import chunked
from multiprocessing import Pool

def make_spec_normalized(filename):   
    rate, data = wavread(filename)
    if data.shape[0] < 16000:
        to_pad = 16000 - data.shape[0]
        start_pad = to_pad // 2
        end_pad = 16000 - data.shape[0] - start_pad
        data = np.pad(data, (start_pad, end_pad), 'reflect')
    elif data.shape[0] >  16000:
        data = data[0:16000]
   
    f, t, xx = spectrogram(data, rate,window=('tukey', .25), nperseg=128, noverlap=64)
    epsilon = 10e-8
    spec_log = np.log10(xx+epsilon)
    spec_log = spec_log / np.max(spec_log)
    
    return spec_log.T[:,:-1]
 
    
    
def data_generator(directory_path):
    # get list of file
    # while loop that permutes the list and returns spectrogram from list
    filenames = [f for f in os.listdir(directory_path)]
    while True:
        random.shuffle(filenames)
        for file in filenames:
            spec = make_spec_normalized(directory_path + file)
            yield spec

def file_generator(directory_path):
    # get list of file
    # while loop that permutes the list and returns spectrogram from list
    filenames = [f for f in os.listdir(directory_path)]
    while True:
        random.shuffle(filenames)
        for file in filenames:
            yield directory_path + file

def batch_generator(root_directory):
    sub_directories = ['left', 'up', 'down', 'no', 'right', 'on', 'yes', 'off', 'unknown', 'silence', 'stop', 'go']
    classes = [v for v in range(len(sub_directories))]
    sub_file_gens = [file_generator(root_directory + sub_dir + '/') for sub_dir in sub_directories]
    
    workers = Pool(4)
    
    while True:
        file_list = [next(gen) for gen in sub_file_gens]
        spec_list = workers.map(make_spec_normalized, file_list)
        yield np.array(spec_list), np.array(classes)
    workers.close()
    workers.join()
        

def mini_batch_generator(path, batch_size):
    a = np.array([0,1,2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)
    ohe = OneHotEncoder()
    ohe.fit(a)
    
    gen = batch_generator(path)
    while True:
        batch_X = []
        batch_Y = []
        for i in range(batch_size):
            X, Y = next(gen)
            Y = ohe.transform(Y.reshape(-1, 1)).toarray()
            batch_X.append(X)
            batch_Y.append(Y)
            
        yield np.vstack(batch_X), np.vstack(batch_Y)        
    
    
def submission_data_generator(filepath, minibatch_size):
    filenames = [f for f in os.listdir(filepath)]
    
    data_chunks = list(chunked(filenames, minibatch_size))
    
    for chunk in data_chunks:
        data = []
        filenames = []
        for file in chunk:
            data.append(make_spec_normalized(filepath + file))
            filenames.append(file)
        yield np.array(data), filenames
    yield None, None 


def time_series_generator(directory_path):
    # get list of file
    # while loop that permutes the list and returns spectrogram from list
    filenames = [f for f in os.listdir(directory_path)]
    while True:
        random.shuffle(filenames)
        for file in filenames:
            rate, data = wavread(directory_path + file)
            if data.shape[0] < 16000:
                to_pad = 16000 - data.shape[0]
                start_pad = to_pad // 2
                end_pad = 16000 - data.shape[0] - start_pad
                data = np.pad(data, (start_pad, end_pad), 'reflect')
            elif data.shape[0] >  16000:
                data = data[0:16000]
                
            yield data.astype(np.float32)
 
def time_series_loader(filepath):
    rate, data = wavread(filepath)
    if data.shape[0] < 16000:
        to_pad = 16000 - data.shape[0]
        start_pad = to_pad // 2
        end_pad = 16000 - data.shape[0] - start_pad
        data = np.pad(data, (start_pad, end_pad), 'reflect')
    elif data.shape[0] >  16000:
        data = data[0:16000]
        
    return data.astype(np.float32)    

           
def time_series_batch_generator(root_directory):
    sub_directories = ['left', 'up', 'down', 'no', 'right', 'on', 'yes', 'off', 'unknown', 'silence', 'stop', 'go']
    classes = [v for v in range(len(sub_directories))]
    file_gens = [file_generator(root_directory + sub_dir + '/') for sub_dir in sub_directories]
    
    while True:
        file_list = [next(gen) for gen in file_gens]
        data_list = map(time_series_loader, file_list)

        yield np.vstack(data_list), np.array(classes)
        
    
def time_series_mini_batch_generator(path, batch_size):
    a = np.array([0,1,2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)
    ohe = OneHotEncoder()
    ohe.fit(a)
    
    gen = time_series_batch_generator(path)
    while True:
        batch_X = []
        batch_Y = []
        for i in range(batch_size):
            X, Y = next(gen)
            Y = ohe.transform(Y.reshape(-1, 1)).toarray().astype(np.float32)
            batch_X.append(X)
            batch_Y.append(Y)
        
        yield np.vstack(batch_X), np.vstack(batch_Y)     
        
    
def class_batch_generator(filepath, cls, batch_size=32):

    data = bcolz.carray(rootdir=filepath+'data')[:]
    num_points = len(data)
    num_splits = num_points // batch_size
    print(cls, num_points)
    while True:
        np.random.shuffle(data)
        for spec in np.array_split(data, num_splits):
            yield spec, np.repeat(cls, len(spec))
    
    
def balanced_batch_generator(base_filepath, batch_size=32):
    classes = ['left', 'up', 'down', 'no', 'right', 'on', 'yes', 'off', 'unknown', 'silence', 'stop', 'go']
    gens = [class_batch_generator(base_filepath + name +'/', cls, batch_size) for cls, name in enumerate(classes) ]
    a = np.array([0,1,2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)
    ohe = OneHotEncoder()
    ohe.fit(a)
    while True:
        batch_x = []
        batch_y = []
        

        for gen in gens:
            X, Y = next(gen)
            batch_x.append(X)
            batch_y.append(Y)
        
        yield np.vstack(batch_x), ohe.transform(np.hstack(batch_y).reshape(-1, 1)).toarray()
        
#        ohe.transform(Y.reshape(-1, 1)).toarray()
    
    
    
if __name__ == '__main__':

#      
    gg = time_series_mini_batch_generator('train/train_test/', 32)

    tt = next(gg)
    
    gg2 = time_series_batch_generator('train/train_test/')

    tt2 = next(gg2)
    tt3 = next(gg2)
    
    vv = np.hstack([tt2[0],tt3[0]])
    
    gen = class_batch_generator('train/bcolz_separate/train/' +'left/' , 0, 256)
    gen2 = class_batch_generator('train/bcolz_separate/train/' +'right/' , 1, 256)
    
    
    x,y = next(gen)
    x2, y2 = next(gen2)

    batch_gen = balanced_batch_generator('train/bcolz_separate/train/', batch_size=32)


    x,y = next(batch_gen)
