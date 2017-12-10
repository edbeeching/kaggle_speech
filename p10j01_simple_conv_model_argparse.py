# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:40:24 2017

@author: Edward
"""
import argparse
import itertools
import random
import os
import bcolz
import numpy as np


from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, Dropout, LSTM, Reshape, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l2

from sklearn.utils import class_weight
  

def load_bcolz_data(filepath):  
    data = bcolz.carray(rootdir=filepath+'data')
    labels = bcolz.carray(rootdir=filepath+'labels')
    
    return np.expand_dims(data[:], axis=3), to_categorical(labels[:])

def create_simple_conv_model(input_shape=(249,64,1), reg=0.0, drop=0.0,scale=False):
    model_input= Input(shape=input_shape)
    if scale:
        x = BatchNormalization()(model_input)
        x = Conv2D(16, kernel_size=(3,3), activation='relu')(x)
    else:
        x = Conv2D(16, kernel_size=(3,3), activation='relu')(model_input)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(32, kernel_size=(1,6),activation='relu')(x)
    x = Reshape((29,32))(x)
    x = LSTM(64, recurrent_regularizer=l2(reg), recurrent_dropout=drop, unroll=True)(x)
    x = Dropout(drop)(x)
    model_output = Dense(12, activation='softmax', kernel_regularizer=l2(reg))(x)
    
    model = Model(model_input, model_output)
    model.summary()
    
    return model

def train_model(input_shape=(249,64,1), reg=0.0, drop=0.0, weight=False, scale=False, max_epoch=50):
      
    if weight:
        weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(Y_train,1)), np.argmax(Y_train,1))  
        print('Using weights')
    else:
        weights = np.ones((12,1))

    model = create_simple_conv_model(input_shape=(249,64,1), reg=reg, drop=drop, scale=scale)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])        
  
    tensor_board = TensorBoard(log_dir='tensorboard/10j01/model_p10j01_simple_conv_model_REG{}_DROP{}_W{}_S{}'.format(reg, drop, weight, scale))
    check_point = ModelCheckpoint(filepath='models/10j01/model_p10j01_simple_conv_model_REG{}_DROP{}_W{}_S{}'.format(reg, drop, weight, scale)+'.{epoch:02d}.hdf5', period=10)

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
    drop_params = [0.0,0.2,0.4,0.6,0.8]
    weight_params = [True, False]
    scaling_params = [True, False]
    num_tests = 20
    
    
    tests = [t for t in itertools.product(reg_params, drop_params, weight_params, scaling_params)]
    random.shuffle(tests)
    
    for test in tests[:num_tests]:
        reg, drop, weight, scale = test
        
        train_model(reg=reg, drop=drop, weight=weight, scale=scale)
        
  