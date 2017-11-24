# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:32:11 2017

@author: Edward
"""

from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l2
import numpy as np
import argparse
import bcolz
import os
  
def load_bcolz_data(filepath):  
    data = bcolz.carray(rootdir=filepath+'data')
    labels = bcolz.carray(rootdir=filepath+'labels')
    
    return np.expand_dims(data[:], axis=3), to_categorical(labels[:])

def create_conv_model1(input_shape=(249,64,1), reg=0.0, drop=0.0):

    model_input = Input(shape=input_shape)
    x = Conv2D(16, kernel_size=(3,3), padding='same', activation='relu')(model_input)
    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Flatten()(x)
    x = Dropout(drop)(x)
    model_output = Dense(12, activation='softmax', kernel_regularizer=l2(0.0001))(x)
    
    model = Model(model_input, model_output)
    model.summary()
    
    return model



model = create_conv_model1()



    parser = argparse.ArgumentParser()
    parser.add_argument('-r', dest="REG", type=float, default=0.0, action='store')
    parser.add_argument('-d', dest="DROP", type=float, default=0.0, action='store')
    parser.add_argument('-e', dest="MAX_EPOCH", type=int, default=0, action='store')    
    
    args = parser.parse_args()
    REG = args.REG
    DROP = args.DROP
    MAX_EPOCH = args.MAX_EPOCH
