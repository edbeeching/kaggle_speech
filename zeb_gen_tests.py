# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:38:52 2017

@author: Edward
"""

neurons = [64, 128, 256, 512]
regs = [0.0,0.0001,0.001,0.01,0.1]


script_name = 'p04j01_simple_lstm_argparse.py'
with open('batcher.sh','w') as f:  
    for neuron in neurons:
        for reg in regs:
            f.write('{} {} {}\n'.format(script_name, neuron, reg))
        
        

