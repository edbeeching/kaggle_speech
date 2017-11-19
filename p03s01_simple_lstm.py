# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 09:10:48 2017

@author: Edward
"""


from data_utils import mini_batch_generator, make_spec_normalized
from keras.models import Model, load_model
    
from sklearn.metrics import confusion_matrix
from keras.layers import LSTM, Dense, Input
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import os
from more_itertools import chunked

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

    SUBMISSION_DATA_PATH = 'test/audio/'
    model = load_model('models/model_p03j01.99.hdf5')

    sub_gen = submission_data_generator(SUBMISSION_DATA_PATH,32)

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
    
    submission_filename = 'submissions/p03s01_simple_lstm_20171119.csv'
    with open(submission_filename, 'w') as f:
        f.write('fname,label\n')
        for k,v in sorted(results_dict.items()):
            f.write('{},{}\n'.format(k, classes[v]))
        
        
    
