# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:08:29 2017

@author: Edward
"""

from data_utils import balanced_batch_generator

from keras.models import Model
from keras.layers import LSTM, Dense, Input
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l2
import numpy as np


def create_model(input_shape, num_neurons=128, regul=0.0, drop=0.0):
    model_input = Input(shape=(input_shape))
    x = LSTM(num_neurons, recurrent_regularizer=l2(regul), recurrent_dropout=drop)(model_input)
    model_output = Dense(12, kernel_regularizer=l2(regul), activation='softmax')(x)
    
    model = Model(model_input, model_output)
    model.summary()    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])  
    return model
    
if __name__ == '__main__':    
    TRAIN_PATH = 'train/bcolz_separate/train/'
    VALID_PATH = 'train/bcolz_separate/valid/'
    
    model = create_model((249,64), 128, 0.0001, 0.5)
    
    tensor_board = TensorBoard(log_dir = 'tensorboard/model_p12j01_simple_lstm_balanced')
    check_point = ModelCheckpoint(filepath='models/model_p12j01_simple_lstm_balanced.{epoch:02d}.hdf5', period=10)

    
    train_generator = balanced_batch_generator(TRAIN_PATH, 32)
    valid_generator = balanced_batch_generator(VALID_PATH, 32)
    

    model.fit_generator(train_generator,
                        steps_per_epoch=60,
                        epochs=100,
                        verbose=1,
                        #callbacks=[tensor_board, check_point],
                        validation_data=valid_generator,
                        validation_steps=10,
                        workers=1,
                        max_queue_size=4,
                        ) 