# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:07:08 2022

@author: Georgia Nixon
"""


import matplotlib.pyplot as plt
import numpy as np

from numpy import pi, exp, sin, cos

import pandas as pd
place = "Georgia Nixon"
import matplotlib as mpl
import seaborn as sns
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import CreateHFGeneral
from hamiltonians import Cosine, ConvertComplex
from hamiltonians import ListRatiosInLowerTriangle
import time
# dataLoc = "/Users/"+place+"/Code/MBQD/floquet-simulations/src/Cluster/Floquet/Data/hpc-work/"
# dataLoc = "E:/Set10-alpha=1,beta=2,rel3phase/"
dataLoc = "E:/Set4-alpha=1,beta=3,A3e{0,5}/"
# dataLoc = "E:/Set4-alpha=1,beta=3,A3e{6,17}/"
# dataLoc = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/Triangle/Cluster/"
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
#%%


alpha = 1
beta = 3

dfs_full = []
dfs_no_raw = []

# folders = [ "Set5"]
# for f in folders:
for A2 in np.linspace(26,30,5):
    
    for A3 in np.linspace(0,30,31):
        # print(str(int(A2)), str(int(A3)))
        
        fileName = "TriangleRatios,alpha="+str(int(alpha))+",beta="+str(int(beta))+",A2="+str(int(A2))+",A3="+str(int(A3))+".csv"
        # fileName = "TriangleRatios,alpha=1,beta=2,A2=0,A3=0.csv"# print(fileName)
        try:
            dfi = pd.read_csv(dataLoc+fileName,
                          index_col=False,
                          dtype = {'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint32,
                              "beta":np.uint32,
                              "phi2/pi":np.float64,
                              "phi3rel/pi":np.float64,
			      "FT-J12-ABS":np.float64,
			      "FT-J23-ABS":np.float64,
			      "FT-J31-ABS":np.float64,
			      "FT-Plaq-PHA":np.float64,
			      "HE-J12-ABS":np.float64,
			      "HE-J23-ABS":np.float64,
			      "HE-J31-ABS":np.float64,
			      "HE-Plaq-PHA":np.float64,
			      "FT-LowerT.X":np.float64,
			      "FT-LowerT.Y":np.float64,
			      "HE-LowerT.X":np.float64,
			      "HE-LowerT.Y":np.float64
                              } )
            print(A2, A3)
            dfi_noraw = dfi.drop(columns=["FT-J12","FT-J23","FT-J31","HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"])
            dfi_noraw = dfi_noraw.astype({'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint32,
                              "beta":np.uint32,
                              "phi2/pi":np.float64,
                              "phi3rel/pi":np.float64,
			      "FT-J12-ABS":np.float64,
			      "FT-J23-ABS":np.float64,
			      "FT-J31-ABS":np.float64,
			      "FT-Plaq-PHA":np.float64,
			      "HE-J12-ABS":np.float64,
			      "HE-J23-ABS":np.float64,
			      "HE-J31-ABS":np.float64,
			      "HE-Plaq-PHA":np.float64,
			      "FT-LowerT.X":np.float64,
			      "FT-LowerT.Y":np.float64,
			      "HE-LowerT.X":np.float64,
			      "HE-LowerT.Y":np.float64
                              })
    
            dfs_no_raw.append(dfi_noraw)
            
        except:
            p = 0
            print("Oh no ", A2, A3)
            


bigData = pd.concat(dfs_no_raw, ignore_index=True)

#FInd rows with nans
nanRows = np.where(np.isnan(ConvertComplex(bigData["FT-J12-ABS"])))[0]
#drop nan locs
print("num of nans:", len(nanRows))
bigData = bigData.drop(nanRows)

st = time.time()    

bigData.to_csv(dataLoc +  "bigData3.csv",
                  index=False, 
                  )
et = time.time()
print("   save took", np.round(et - st, 1), "s")


#%%
#import
st = time.time()    

df1 = pd.read_csv(dataLoc+"Set4Phases.csv", index_col=False)
# df1 = pd.read_csv(dataLoc+"Set4Summary.csv", index_col=False)
# df2 = pd.read_csv(dataLoc+"bigData2.csv", index_col=False)
# df3 = pd.read_csv(dataLoc+"bigData3.csv", index_col=False)

et = time.time()

print("   import took", np.round(et - st, 1), "s")

#%%
# omegas = 1601
# A2s = 31
# A3s = 6
# phi3/pi = 41


#%%
bigData = pd.concat([df1,df2,df3], ignore_index=True)

st = time.time()    

df1.to_csv(dataLoc + "Test.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")

#%%

df1 = df1.rename(columns={"FT-J31-PHA": "FT-Plaq-PHA", "HE-J31-PHA": "HE-Plaq-PHA"})

df1 = df1.astype({'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint8,
                              "beta":np.uint8,
                              "phi3/pi":np.float64,
                              # "phi3rel/pi":np.float64,
			      "FT-J12-ABS":np.float64,
			      "FT-J23-ABS":np.float64,
			      "FT-J31-ABS":np.float64,
			      "FT-Plaq-PHA":np.float64,
			      "HE-J12-ABS":np.float64,
			      "HE-J23-ABS":np.float64,
			      "HE-J31-ABS":np.float64,
			      "HE-Plaq-PHA":np.float64,
			      "FT-LowerT.X":np.float64,
			      "FT-LowerT.Y":np.float64,
			      "HE-LowerT.X":np.float64,
			      "HE-LowerT.Y":np.float64
                              })

#%%
# save phases data only
dfPha = df1.drop(columns=[ 'FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'HE-J12-ABS', 'HE-J23-ABS',
       'HE-J31-ABS',  'FT-LowerT.X', 'FT-LowerT.Y',
       'HE-LowerT.X', 'HE-LowerT.Y'] )

st = time.time()    
dfPha.to_csv(dataLoc + "Phases.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")

#%%
# save lower triangle data only
dfLowerT = df1.drop(columns=['FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'FT-Plaq-PHA', 'HE-J12-ABS', 'HE-J23-ABS',
       'HE-J31-ABS', 'HE-Plaq-PHA' ] )
st = time.time()    
dfLowerT.to_csv(dataLoc + "LowerTriangle.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


#%%
# save HT data only
dfHE = df1.drop(columns=['FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'FT-Plaq-PHA', 'FT-LowerT.X', 'FT-LowerT.Y' ] )
st = time.time()    
dfHE.to_csv(dataLoc + "Set4HE.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


#%%
# save HT data only
dfFT = df1.drop(columns=['HE-J12-ABS',
       'HE-J23-ABS', 'HE-J31-ABS', 'HE-Plaq-PHA', 'HE-LowerT.X', 'HE-LowerT.Y' ] )
st = time.time()    
dfFT.to_csv(dataLoc + "Set4FT.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")
