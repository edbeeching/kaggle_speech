# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:31:32 2017

@author: Edward
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
from librosa.feature import melspectrogram as mfcc

def loadmfcc(filepath):
    rate, data = wavread(filepath)
    mels = mfcc(data, rate, n_fft=512, hop_length=256)
    return mels[0:40,:]


CLASSES = 'yes, no, up, down, left, right, on, off, stop, go, silince, unknown'.split()
TRAIN_BASE = 'train/audio/'
DIRECTORIES =  [d for d in os.listdir(TRAIN_BASE)]

DIR0 = [f for f in os.listdir(TRAIN_BASE + DIRECTORIES[0])]
FILE0_DIR0 = DIR0[0]

rate, data = wavread(TRAIN_BASE + DIRECTORIES[0] + '/' + FILE0_DIR0)


#plt.specgram(data, Fs=rate)

m = loadmfcc(TRAIN_BASE + DIRECTORIES[0] + '/' + FILE0_DIR0)
plt.imshow(m)




#
#mels = mfcc(data, rate, n_fft=512, hop_length=256)
#plt.figure()
#plt.imshow(mels[0:40,:])
#
#with open('meltest.npy', 'wb') as f:
#    np.save(f,mels)
#    
#mels2 = None
#with open('meltest.npy', 'rb') as f:
#    mels2 = np.load(f)
    
    

    
    
    
    
    