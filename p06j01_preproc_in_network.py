# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:23:50 2017

@author: Edward
"""

from preproc_utils import Spectrogram, Normalization2D
from data_utils import time_series_mini_batch_generator as batch_gen
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Input, Permute, LSTM, Dense, Reshape, CuDNNLSTM
from keras.callbacks import TensorBoard, ModelCheckpoint
import argparse

def create_stft_model(input_shape=(1,16000), num_neurons=128, regul=0.0):
    model_input = Input(shape=input_shape)
    spec = Spectrogram(n_dft=128, n_hop=64, 
          return_decibel_spectrogram=True, power_spectrogram=1.0, image_data_format='channels_last', 
          trainable_kernel=False, name='static_stft')(model_input)
    norm = Normalization2D(int_axis=1)(spec)
    x = Reshape((-1,250))(norm)
    permuted = Permute((2,1))(x)
    lstm = CuDNNLSTM(num_neurons)(permuted)
    model_output = Dense(12,kernel_regularizer=l2(regul), activation='softmax')(lstm)

    model = Model(model_input, model_output)
    model.summary()
    return model


if __name__ == '__main__':
    
    TRAIN_PATH = 'train/train_train/'
    VALID_PATH = 'train/train_valid/'


    parser = argparse.ArgumentParser()
    parser.add_argument("NUM_NEURONS", type=int, default=128)
    parser.add_argument("REG", type=float, default=0.0)
    #parser.add_argument("WEIGHTS", type=bool, default=False, action='store_true')
    args = parser.parse_args()

    NUM_NEURONS = args.NUM_NEURONS
    REG = args.REG
    #WEIGHTS = args.WEIGHTS

    print('N={}, R={}'.format(NUM_NEURONS, REG))



    model = create_stft_model((1,16000),NUM_NEURONS, REG)
        
    tensor_board = TensorBoard(log_dir = 'tensorboard/model_p04j02_simple_lstm_bi_argparse_N{}_R{}'.format(NUM_NEURONS, REG))
    check_point = ModelCheckpoint(filepath='models/model_p04j02_simple_lstm_bi_argparse_N{}_R{}_'.format(NUM_NEURONS, REG) + '.{epoch:02d}.hdf5', period=10)
   
    train_generator = batch_gen(TRAIN_PATH, 32)
    valid_generator = batch_gen(VALID_PATH, 32)
    
       
    model.fit_generator(train_generator,
                        steps_per_epoch=60,
                        epochs=100,
                        verbose=1,
                        callbacks=[tensor_board, check_point],
                        validation_data=valid_generator,
                        validation_steps=10,
                        workers=1,
                        max_queue_size=10,
                        )    
    
    
    