# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:06:33 2017

@author: Edward
"""

from data_utils import submission_data_generator
from keras.models import load_model
import numpy as np
import argparse


if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument("NUM_NEURONS", type=int, default=128)
    parser.add_argument("REG", type=float, default=0.0)
    parser.add_argument("DROP", type=float, default=0.0)
    #parser.add_argument("WEIGHTS", type=bool, default=False, action='store_true')
    args = parser.parse_args()

    NUM_NEURONS = args.NUM_NEURONS
    REG = args.REG
    DROP = args.DROP
    #WEIGHTS = args.WEIGHTS

    print('N={}, R={}, D={}'.format(NUM_NEURONS, REG, DROP))

    SUBMISSION_DATA_PATH = 'test/audio/'
    sub_gen = submission_data_generator(SUBMISSION_DATA_PATH, 32)
    model = load_model('models/model_p05j02_simple_lstm_rec_reg_argparse_N{}_R{}_D{}.100.hdf5'.format(NUM_NEURONS, REG, DROP))
    
    counter = 0
    results_dict = {}

    while True:
        X, F = next(sub_gen)
        if X == None: break
        if counter % 1600 == 0:
            print(counter)
        preds = model.predict(X)    
        maxes = np.argmax(preds,1)
        for file, pred in zip(F, maxes):
            results_dict[file] = pred
        counter += 32

    classes = ['left', 'up', 'down', 'no', 'right', 'on', 'yes', 'off', 'unknown', 'silence', 'stop', 'go']
    
    submission_filename = 'submissions/p05s02_simple_lstm_rec_reg_argparse_N{}_R{}_D{}_100_20171121_1.csv'.format(NUM_NEURONS, REG, DROP)
    
    with open(submission_filename, 'w') as f:
        f.write('fname,label\n')
        for k,v in sorted(results_dict.items()):
            f.write('{},{}\n'.format(k, classes[v]))  
    
    
