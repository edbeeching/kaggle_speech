# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 10:27:55 2017

@author: Edward
"""

import os 
import numpy as np
import random
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
   
    f, t, xx = spectrogram(data, rate,window=('tukey', .25), nperseg=128)
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
    
    workers = Pool(12)
    
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

            

if __name__ == '__main__':
#    path = 'train/train_valid/bird/'     
#    
#    files = [f for f in os.listdir(path)]
#    
#    gener = data_generator('train/train_valid/bird/')
#    
#    sp = next(gener)
#    
#    sps = []
#    
#    for i in range(10):
#        sps.append(next(gener))
#    
#    plt.figure()
#    for i, sp in enumerate(sps, 1):
#        plt.subplot(2,5,i)
#        plt.imshow(sp)
    
    gener = mini_batch_generator('train/train_test/', 32)
    
    specs, classes = next(gener)
#    plt.figure()
#    for (sp, cl) in zip(specs, classes):
#        plt.subplot(3,4,cl+1)
#        plt.title(str(cl))
#        plt.imshow(sp)
#        

    
    
    
    