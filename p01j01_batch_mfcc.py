# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:02:30 2017

@author: Edward

Program to batch the wav files to MF

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


