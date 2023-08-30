# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 09:49:43 2022

@author: Dell
"""

import os
import pandas as pd
path = r'F:\H1'
d1=[]
for filename in filenames:
    d1.append(pd.read_csv(filename))
df=pd.concat(d1)
df.to_csv('pred.csv')












