# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:16:12 2017

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
from sklearn.utils import class_weight
  

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
    
if __name__ == '__main__': 
       
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', dest="REG", type=float, default=0.0, action='store')
    parser.add_argument('-d', dest="DROP", type=float, default=0.0, action='store')
    parser.add_argument('-e', dest="MAX_EPOCH", type=int, default=10, action='store') 
    parser.add_argument('-w', dest="WEIGHTS", type=bool, default=False, action='store')
    
    args = parser.parse_args()
    REG = args.REG
    DROP = args.DROP
    MAX_EPOCH = args.MAX_EPOCH
    WEIGHTS = args.WEIGHTS

    print('R={}, D={}, E={}, W={}'.format(REG, DROP, MAX_EPOCH, WEIGHTS))
    
    

    
    BCOLZ_TRAIN_PATH = 'train/bcolz/train/'
    BCOLZ_VALID_PATH = 'train/bcolz/valid/'    
    X_train, Y_train = load_bcolz_data(BCOLZ_TRAIN_PATH)
    X_valid, Y_valid = load_bcolz_data(BCOLZ_VALID_PATH)
    
    if WEIGHTS:
        weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(Y_train,1)), np.argmax(Y_train,1))  
        print('Using weights')
    else:
        weights = np.ones(12,1)
    # load latest model if it already exists
    all_models = [model for model in os.listdir('models/') if 
                  'model_p07j02_conv1_REG{}_DROP{}_W{}'.format(REG, DROP, WEIGHTS) in model]
    
    initial_epoch = 0
    model = None
    if len(all_models) > 0:
        model = load_model('models/'+ all_models[-1])
        initial_epoch = int(all_models[-1].split('.')[-2])
    else:
        model = create_conv_model1(input_shape=(249,64,1), reg=REG, drop=DROP)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])        

    
    tensor_board = TensorBoard(log_dir='tensorboard/model_p07j02_conv1_REG{}_DROP{}_W{}'.format(REG, DROP, WEIGHTS))
    check_point = ModelCheckpoint(filepath='models/model_p07j02_conv1_REG{}_DROP{}_W{}'.format(REG, DROP, WEIGHTS)+'.{epoch:02d}.hdf5', period=10)


    model.fit(X_train, Y_train, 
              batch_size=64, epochs=MAX_EPOCH, initial_epoch=initial_epoch,
              validation_data=(X_valid, Y_valid), 
              callbacks=[tensor_board, check_point], 
              verbose=1, class_weight=weights)

    
    
    