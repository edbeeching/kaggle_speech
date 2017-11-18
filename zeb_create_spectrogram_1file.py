# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:01:04 2017

@author: Edward
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
from librosa.feature import melspectrogram as mfcc
from scipy.signal import spectrogram
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


plt.specgram(data, Fs=rate)

f, t, xx = spec(data, rate,window=('tukey',.25),nperseg=128)
plt.figure()
plt.imshow(np.log10(xx))

spec_log = np.log10(xx)

sp_mean = np.mean(spec_log, 1)
sp_std = np.std(spec_log, 1)

spec_log_norm = ((spec_log.T - sp_mean.T)/ sp_std.T).T # annoying due to trailing dim

plt.figure()
plt.imshow(spec_log_norm)
