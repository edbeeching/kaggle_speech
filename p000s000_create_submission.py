# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:46:59 2017

@author: Edward
"""


import datetime
import argparse
import numpy as np
from keras.models import load_model
from data_utils import submission_data_generator


if __name__ == '__main__': 
       
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest="MODEL_NAME", type=str, default='', action='store')

    args = parser.parse_args()
    MODEL_NAME = args.MODEL_NAME

    print('model={}'.format(MODEL_NAME))
       
    SUBMISSION_DATA_PATH = 'test/audio/'
    sub_gen = submission_data_generator(SUBMISSION_DATA_PATH, 32)
    model = load_model('models/'+ MODEL_NAME)
    
    counter = 0
    results_dict = {}

    while True:
        X, F = next(sub_gen)
        if X is None: break
        if counter % 1600 == 0:
            print(counter)
        preds = model.predict(np.expand_dims(X,axis=3))    
        maxes = np.argmax(preds,1)
        for file, pred in zip(F, maxes):
            results_dict[file] = pred
        counter += 32

    classes = ['left', 'up', 'down', 'no', 'right', 'on', 'yes', 'off', 'unknown', 'silence', 'stop', 'go']
    
    now = datetime.datetime.now()
    submission_filename = 'submissions/{}_{}{}{}.csv'.format(MODEL_NAME, now.year, now.month, now.day)
    
    with open(submission_filename, 'w') as f:
        f.write('fname,label\n')
        for k,v in sorted(results_dict.items()):
            f.write('{},{}\n'.format(k, classes[v]))  
            
