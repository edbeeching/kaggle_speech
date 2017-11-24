# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:26:18 2017

@author: Edward
"""

import os
import argparse
import numpy as np
from keras.models import load_model
from data_utils import submission_data_generator


if __name__ == '__main__': 
       
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',dest="REG", type=float, default=0.0, action='store')
    parser.add_argument('-d',dest="DROP", type=float, default=0.0, action='store')
    parser.add_argument('-e',dest="EPOCH", type=float, default=0.0, action='store')

    args = parser.parse_args()
    REG = args.REG
    DROP = args.DROP

    print('R={}, D={}'.format(REG, DROP))
       
    SUBMISSION_DATA_PATH = 'test/audio/'
    sub_gen = submission_data_generator(SUBMISSION_DATA_PATH, 32)
    all_models = [model for model in os.listdir('models/') if 'model_p07j02_conv1_REG{}_DROP{}'.format(REG, DROP) in model]
    model = load_model('models/'+ all_models[-1])
    
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
    
    submission_filename = 'submissions/p07s02_conv1_argparse_R{}_D{}_30_20171123_2.csv'.format(REG, DROP)
    
    with open(submission_filename, 'w') as f:
        f.write('fname,label\n')
        for k,v in sorted(results_dict.items()):
            f.write('{},{}\n'.format(k, classes[v]))  
            
