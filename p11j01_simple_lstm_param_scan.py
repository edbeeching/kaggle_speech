# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:46:03 2017

@author: Edward
"""



import itertools
import random
import bcolz
import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, Dropout, LSTM
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l2

from sklearn.utils import class_weight
  

def load_bcolz_data(filepath):  
    data = bcolz.carray(rootdir=filepath+'data')
    labels = bcolz.carray(rootdir=filepath+'labels')
    
    return data[:], to_categorical(labels[:])

def create_simple_lstm_model(input_shape, num_neurons=256, reg=0.0, drop=0.0):
    model_input = Input(shape=(input_shape))
    x = LSTM(num_neurons, recurrent_regularizer=l2(reg), recurrent_dropout=drop, unroll=True)(model_input)
    model_output = Dense(12, kernel_regularizer=l2(reg), activation='softmax')(x)
    
    model = Model(model_input, model_output)
    model.summary()     
    
    return model

def train_model(input_shape=(249,64,1), reg=0.0, drop=0.0, num_neurons=256, weight=False, max_epoch=100):
       
    if weight:
        weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(Y_train,1)), np.argmax(Y_train,1))  
        print('Using weights')
    else:
        weights = np.ones((12,1))

    model = create_simple_lstm_model(input_shape=(249,64), num_neurons=num_neurons, reg=reg, drop=drop)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])        

    tensor_board = TensorBoard(log_dir='tensorboard/11j01/model_p11j01_simple_lstm_param_scan_N{}_REG{}_DROP{}_W{}'.format(num_neurons,reg, drop, weight))
    check_point = ModelCheckpoint(filepath='models/11j01/model_p11j01_simple_lstm_param_scan_model_N{}_REG{}_DROP{}_W{}'.format(num_neurons, reg, drop, weight)+'.{epoch:02d}.hdf5', period=10)

    model.fit(X_train, Y_train, 
              batch_size=512, epochs=max_epoch,
              validation_data=(X_valid, Y_valid), 
              callbacks=[tensor_board, check_point], 
              verbose=1, class_weight=weights)
    
    
    
if __name__ == '__main__': 
        
    BCOLZ_TRAIN_PATH = 'train/bcolz/train/'
    BCOLZ_VALID_PATH = 'train/bcolz/valid/'    
    X_train, Y_train = load_bcolz_data(BCOLZ_TRAIN_PATH)
    X_valid, Y_valid = load_bcolz_data(BCOLZ_VALID_PATH)
    
    reg_params = [0.0] + [10**x for x in range(-4,0)]    
    drop_params = [0.0, 0.2, 0.4, 0.6, 0.8]
    neuron_params = [64, 128, 256, 512]
    weight_params = [True, False]
    num_tests = 20
        
    tests = [t for t in itertools.product(reg_params, drop_params, neuron_params, weight_params)]
    random.shuffle(tests)
    
    for test in tests[:num_tests]:
        reg, drop, neurons, weight = test
        train_model(reg=reg, drop=drop,num_neurons=neurons, weight=weight)  
    
    
    
    