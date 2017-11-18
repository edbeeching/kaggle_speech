# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:17:26 2017

@author: Edward
"""
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
import os
import random
INPUT_PATH = 'train/audio/_background_noise_/'
TRAIN_PATH = 'train/train_train/silence/'
VALID_PATH = 'train/train_valid/silence/'
TEST_PATH = 'train/train_test/silence/'

files = [f for f in os.listdir(INPUT_PATH) if f[-4:] == '.wav']

samples = []

TRAIN_MIN = 0.7
VALID_MIN = 0.85
TEST_MIN = 1.0

for f in files:
    rate, data = wavread(INPUT_PATH + f)
    
    step = 16000
    num_samples = len(data)
    frames = []
    for i in range(0,num_samples-step, step):        
        frames.append(data[i:i+step])
    
    for i, frame in enumerate(frames, 1):
        val = random.uniform(0.0,1.0)
        if val < TRAIN_MIN:
            wavwrite(TRAIN_PATH+'{}_{:003}.wav'.format(f[:-4],i), rate, frame)           
        elif val < VALID_MIN:
            wavwrite(VALID_PATH+'{}_{:003}.wav'.format(f[:-4],i), rate, frame)
        else:
            wavwrite(TEST_PATH+'{}_{:003}.wav'.format(f[:-4],i), rate, frame)
            