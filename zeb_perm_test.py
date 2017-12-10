# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:06:42 2017

@author: Edward
"""

import more_itertools
import itertools
import random

a = [1,2,3,4,5]
b = 'a b c d e'.split()

tests = [t for t in itertools.product(a,b)]

for t in tests:
    print(t)
random.shuffle(tests)
for t in tests :
    print(t)

    


import datetime
now = datetime.datetime.now()
print(now.year, now.month, now.day)
