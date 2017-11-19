# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 07:36:21 2017

@author: Edward
"""


from data_utils import mini_batch_generator
from keras.models import load_model
  
from sklearn.metrics import confusion_matrix
import numpy as np




if __name__ == '__main__':

    TRAIN_PATH = 'train/train_train/'
    VALID_PATH = 'train/train_valid/'
    TEST_PATH = 'train/train_test/'

    model = load_model('models/model_p03j02.199.hdf5')


    test_generator =  mini_batch_generator(TEST_PATH, 32)
    
    
    Yt = []
    Yp = []
    
    for i in range(10):
        X,Y = next(test_generator)
        preds = model.predict(X)    
    
        Yt.append(np.argmax(Y,1))
        Yp.append(np.argmax(preds,1))
    
    cm = confusion_matrix(np.hstack(Yt), np.hstack(Yp))
