# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:06:33 2017

@author: Edward
"""

from data_utils import submission_data_generator
from keras.models import load_model
import numpy as np


    
if __name__ == '__main__':

    SUBMISSION_DATA_PATH = 'test/audio/'
    model = load_model('models/model_p04j01_simple_lstm_argparse_N512_R0.0001_.89.hdf5')

    sub_gen = submission_data_generator(SUBMISSION_DATA_PATH, 32)

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
    
    submission_filename = 'submissions/p04s02_simple_lstm_N512_R0p0001_epoch89_20171120_1.csv'
    
    with open(submission_filename, 'w') as f:
        f.write('fname,label\n')
        for k,v in sorted(results_dict.items()):
            f.write('{},{}\n'.format(k, classes[v]))
 
    
    
