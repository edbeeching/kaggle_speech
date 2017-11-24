# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:55:13 2017

@author: Edward
"""

from data_utils import mini_batch_generator
from data_utils import make_spec_normalized
from keras.models import Model
from keras.layers import LSTM, Dense, Input, CuDNNLSTM, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l2
import numpy as np
import argparse
import os
import bcolz
  
def load_bcolz_data(filepath):  
    data = bcolz.carray(rootdir=filepath+'data')
    labels = bcolz.carray(rootdir=filepath+'labels')
    
    return np.expand_dims(data[:], axis=3), to_categorical(labels[:])

def create_conv_model1(input_shape=(249,64,1)):

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
    model_output = Dense(12, activation='softmax', kernel_regularizer=l2(0.0001))(x)
    
    model = Model(model_input, model_output)
    model.summary()
    
    return model
    
if __name__ == '__main__': 
    
    BCOLZ_TRAIN_PATH = 'train/bcolz/train/'
    BCOLZ_VALID_PATH = 'train/bcolz/valid/'
    
    
    X_train, Y_train = load_bcolz_data(BCOLZ_TRAIN_PATH)
    X_valid, Y_valid = load_bcolz_data(BCOLZ_VALID_PATH)
    
    model = create_conv_model1()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    
    tensor_board = TensorBoard(log_dir = 'tensorboard/model_p07j01_conv')
    check_point = ModelCheckpoint(filepath='models/model_p07j01_conv.{epoch:02d}.hdf5', period=10)


    model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_data=(X_valid, Y_valid), callbacks=[tensor_board, check_point],verbose=1)

    
    
    