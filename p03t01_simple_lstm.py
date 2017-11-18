# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:45:51 2017

@author: Edward
"""

from data_utils import batch_generator

from keras.models import Model
from keras.layers import LSTM, Dense, Input
from keras.utils import to_categorical
from keras.regularizers import l2
import numpy as np

#def create_model(input_size):
#    
#    model_input = Input(input_size)
#    x = LSTM()(model_input)
    
    

TRAIN_PATH = 'train/train_train/'
VALID_PATH = 'train/train_valid/'
def mini_batch_generator(path, batch_size):
    gen = batch_generator(TRAIN_PATH)
    while True:
        batch_X = []
        batch_Y = []
        for i in range(batch_size):
            X, Y = next(gen)
            Y = to_categorical(Y)
            batch_X.append(X)
            batch_Y.append(Y)
            
        yield np.vstack(batch_X), np.vstack(batch_Y)
   
    
train_generator = mini_batch_generator(TRAIN_PATH, 32)
valid_generator = mini_batch_generator(VALID_PATH, 32)
X, Y = next(train_generator)
Xv, Yv = next(valid_generator)

model_input = Input(shape=(142,64))
x = LSTM(128)(model_input)
model_output = Dense(12, activation='softmax')(x)

model = Model(model_input, model_output)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

model.fit(X,Y,epochs=100,validation_data=(Xv,Yv))

Y_preds = model.predict(X)
Yv_preds = model.predict(Xv)

from sklearn.metrics import confusion_matrix

cm_train = confusion_matrix(np.argmax(Y,1), np.argmax(Y_preds,1))
cm_valid = confusion_matrix(np.argmax(Yv,1), np.argmax(Yv_preds,1))



