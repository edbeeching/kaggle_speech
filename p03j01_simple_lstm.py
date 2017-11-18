# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:06:33 2017

@author: Edward
"""

from data_utils import mini_batch_generator

from keras.models import Model
from keras.layers import LSTM, Dense, Input
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np



def create_model(input_shape):
    model_input = Input(shape=(input_shape))
    x = LSTM(128)(model_input)
    model_output = Dense(12, activation='softmax')(x)
    
    model = Model(model_input, model_output)
    model.summary()    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])  
    
    return model

if __name__ == '__main__':

    TRAIN_PATH = 'train/train_train/'
    VALID_PATH = 'train/train_valid/'

    model = create_model((142,64))
    
    tensor_board = TensorBoard(log_dir = 'tensorboard/model_p03j01/')
    check_point = ModelCheckpoint(filepath='models/model_p03j01.{epoch:02d}.hdf5', period=10)

    
    train_generator = mini_batch_generator(TRAIN_PATH, 32)
    valid_generator = mini_batch_generator(VALID_PATH, 32)
    

    model.fit_generator(train_generator,
                        steps_per_epoch=60,
                        epochs=100,
                        verbose=1,
                        callbacks=[tensor_board, check_point],
                        validation_data=valid_generator,
                        validation_steps=10,
                        workers=1,
                        max_queue_size=4,
                        )    
    
    
