# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:51:24 2022

@author: Georgia
"""

import pandas as pd
import numpy as np
import time

# dataLocA = "D:/Data/Set12-alpha=1,beta=2,omega=8/"
dfA = pd.read_csv("D:/Data/Set12-alpha=1,beta=2,omega=8/"+
                  "Summaries/HE-Min.csv",
                          index_col=False)
dfB = pd.read_csv("D:/Data/Set16-alpha=1,beta=2,omega=8,+.13/"+
                  "Summaries/HE-Min.csv",
                          index_col=False)
dfC = pd.read_csv("D:/Data/Set15-alpha=1,beta=2,omega=8,+.24/"+
                  "Summaries/HE-Min.csv",
                          index_col=False)

dfO = pd.concat([dfA, dfB, dfC], ignore_index=True)

#
st = time.time()    
dfO.to_csv("D:/Data/Merges/alpha=1,beta=2,omega=8/HE-Min.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


#%%