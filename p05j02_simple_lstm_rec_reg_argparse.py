# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:06:33 2017

@author: Edward
"""

from data_utils import mini_batch_generator

from keras.models import Model
from keras.layers import LSTM, Dense, Input, CuDNNLSTM
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l2
import numpy as np
import argparse


def create_model(input_shape, num_neurons=128, regul=0.0, drop=0.0):
    model_input = Input(shape=(input_shape))
    x = LSTM(num_neurons, recurrent_regularizer=l2(regul), recurrent_dropout=drop)(model_input)
    model_output = Dense(12, kernel_regularizer=l2(regul), activation='softmax')(x)
    
    model = Model(model_input, model_output)
    model.summary()    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])  
    
    return model

if __name__ == '__main__':    
    TRAIN_PATH = 'train/train_train/'
    VALID_PATH = 'train/train_valid/'

    parser = argparse.ArgumentParser()
    parser.add_argument("NUM_NEURONS", type=int, default=128)
    parser.add_argument("REG", type=float, default=0.0)
    parser.add_argument("DROP", type=float, default=0.0)
    #parser.add_argument("WEIGHTS", type=bool, default=False, action='store_true')
    args = parser.parse_args()

    NUM_NEURONS = args.NUM_NEURONS
    REG = args.REG
    DROP = args.DROP
    #WEIGHTS = args.WEIGHTS

    print('N={}, R={}, D={}'.format(NUM_NEURONS, REG, DROP))

    
    model = create_model((249,64), NUM_NEURONS, REG, DROP)
    
    tensor_board = TensorBoard(log_dir = 'tensorboard/model_p05j02_simple_lstm_rec_reg_argparse_N{}_R{}_D{}'.format(NUM_NEURONS, REG,DROP))
    check_point = ModelCheckpoint(filepath='models/model_p05j02_simple_lstm_rec_reg_argparse_N{}_R{}_D{}'.format(NUM_NEURONS, REG,DROP) + '.{epoch:02d}.hdf5', period=10)

    
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
    
    
