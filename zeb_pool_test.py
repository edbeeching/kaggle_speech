# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:59:14 2017

@author: Edward
"""

from multiprocessing import Pool
import random
import time
def square(x):
    time.sleep(random.random())
    return x*x

if __name__ == '__main__':
    p = Pool(5)
    
    results = p.map(square,[1,2,3,4,5,6,7,8,9])
    
    print(results)
    
    
    
    