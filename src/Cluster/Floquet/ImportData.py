# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:07:08 2022

@author: Georgia Nixon
"""


import matplotlib.pyplot as plt
import numpy as np

from numpy import pi, exp, sin, cos

import pandas as pd
place = "Georgia"
import matplotlib as mpl
import seaborn as sns
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import CreateHFGeneral
from hamiltonians import Cosine, ConvertComplex
from hamiltonians import ListRatiosInLowerTriangle
import time



def ConvertComplex(s):
    """For retrieving complex numbers from csv's"""
    return np.complex128(s.replace('i', 'j').replace('*I', 'j').replace('*^', 'e'))


dataLoc = "D:/Data/Set20/"

#%%


alpha = 1
beta = 2


ohnos = []

dfs_no_raw = []

for A2 in np.linspace(0,40,41):
    for A3 in np.linspace(0,40,41):
        # print(str(int(A2)), str(int(A3)))
        
        fileName = ("Raw/TriangleRatios,alpha="+str(int(alpha))
                    +",beta="+str(int(beta))+",A2="+str(int(A2))
                    +",A3="+str(int(A3))+".csv")
        # fileName = "TriangleRatios,alpha=1,beta=2,A2=0,A3=0.csv"# print(fileName)
        try:
            dfi = pd.read_csv(dataLoc+fileName,
                          index_col=False,
                          dtype = {'A2': np.float64,
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
                              },
                #           converters={
                # 'FT-J12':ConvertComplex,
                # 'FT-J23':ConvertComplex,
                # 'FT-J31':ConvertComplex, 
                # 'HE-J12':ConvertComplex,
                # 'HE-J23':ConvertComplex,
                # 'HE-J31':ConvertComplex,
                # "HE-O1":ConvertComplex,
                # "HE-O2":ConvertComplex,
                # "HE-O3":ConvertComplex
                #               }
                          )
            
            print(A2, A3)
            # dfi = dfi.drop(columns=["FT-J12","FT-J23","FT-J31","HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"])
            
            #FInd rows with nans
            nanRows = np.where(np.isnan(dfi["FT-J12-ABS"]))[0]
            #drop nan locs
            print("  # nans:", len(nanRows))
            dfi = dfi.drop(nanRows)
            
            for col in ["FT-J12","FT-J23","FT-J31","HE-J12",
                        "HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"]:
                dfi[col] = ConvertComplex(dfi[col])
            
          #   dfi = dfi.astype({'A2': np.float64,
          #                     'A3': np.float64,
          #                     'omega0': np.float64,
          #                     "alpha":np.uint8,
          #                     "beta":np.uint8,
          #                     "phi3/pi":np.float64,
          #                     # "phi3rel/pi":np.float64,
 			      # "FT-J12-ABS":np.float64,
 			      # "FT-J23-ABS":np.float64,
 			      # "FT-J31-ABS":np.float64,
 			      # "FT-Plaq-PHA":np.float64,
 			      # "HE-J12-ABS":np.float64,
 			      # "HE-J23-ABS":np.float64,
 			      # "HE-J31-ABS":np.float64,
 			      # "HE-Plaq-PHA":np.float64,
 			      # "FT-LowerT.X":np.float64,
 			      # "FT-LowerT.Y":np.float64,
 			      # "HE-LowerT.X":np.float64,
 			      # "HE-LowerT.Y":np.float64,

          #       # 'FT-J12':np.complex128,
          #       # 'FT-J23':np.complex128,
          #       # 'FT-J31':np.complex128,
          #       # 'HE-J12':np.complex128,
          #       # 'HE-J23':np.complex128,
          #       # 'HE-J31':np.complex128,
          #       # "HE-O1":np.complex128,
          #       # "HE-O2":np.complex128,
          #       # "HE-O3":np.complex128

          #                     })
    
            dfs_no_raw.append(dfi)
            
        except:
            p = 0
            print("Oh no ", A2, A3)
            ohnos.append([A2, A3])
            


bigData = pd.concat(dfs_no_raw, ignore_index=True)



st = time.time()    
bigData.to_csv(dataLoc +  "Summaries/Summary.csv",
                  index=False, 
                  )
et = time.time() 
print("   save took", np.round(et - st, 1), "s")

# bigData = bigData.drop(columns=["FT-J12","FT-J23","FT-J31","HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"])


# save phases data only
dfPha = bigData.drop(columns=[ 'FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'HE-J12-ABS', 'HE-J23-ABS',
       'HE-J31-ABS',  'FT-LowerT.X', 'FT-LowerT.Y',
       'HE-LowerT.X', 'HE-LowerT.Y',
       "FT-J12","FT-J23","FT-J31","HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"] )

st = time.time()    
dfPha.to_csv(dataLoc + "Summaries/Phases.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


# save lower triangle data only
dfLowerT = bigData.drop(columns=['FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'FT-Plaq-PHA', 'HE-J12-ABS', 'HE-J23-ABS',
       'HE-J31-ABS', 'HE-Plaq-PHA',
       "FT-J12","FT-J23","FT-J31","HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"] )
st = time.time()    
dfLowerT.to_csv(dataLoc + "Summaries/LowerTriangle.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


# save HT data only
dfHE = bigData.drop(columns=['FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'FT-Plaq-PHA', 'FT-LowerT.X', 'FT-LowerT.Y',
       "FT-J12","FT-J23","FT-J31"] )
st = time.time()    
dfHE.to_csv(dataLoc + "Summaries/HE-Raw.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


dfHE = dfHE.drop(columns=["HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"])
st = time.time()    
dfHE.to_csv(dataLoc + "Summaries/HE.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")




dfHE = dfHE.drop(columns = ['HE-J12-ABS',
       'HE-J23-ABS', 'HE-J31-ABS'])
st = time.time()    
dfHE.to_csv(dataLoc + "Summaries/HE-Min.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


# save FT data only
dfFT = bigData.drop(columns=['HE-J12-ABS',
       'HE-J23-ABS', 'HE-J31-ABS', 'HE-Plaq-PHA', 'HE-LowerT.X', 'HE-LowerT.Y',
       "HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"] )
st = time.time()    
dfFT.to_csv(dataLoc + "Summaries/FT-Raw.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


dfFT = dfFT.drop(columns=[ "FT-J12","FT-J23","FT-J31"])
st = time.time()    
dfFT.to_csv(dataLoc + "Summaries/FT.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


dfFT = dfFT.drop(columns = ['FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS'])

st = time.time()    
dfFT.to_csv(dataLoc + "Summaries/FT-Min.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")





#%%

"""Merge data"""

calc_type = "HE"
import pandas as pd
import numpy as np
import time
df1 = pd.read_csv("D:/Data/Merges/alpha=1,beta=2,omega=8,0-30/"+calc_type+"/"
                  +calc_type+"-Min.csv",index_col=False)
df2 = pd.read_csv("D:/Data/Set19-alpha=1,beta=2,omega=8,3040,even/Summaries/"
                  +calc_type+"-Min.csv",index_col=False)
df3 = pd.read_csv("D:/Data/Set20-alpha=1,beta=2,omega=8,3040,odd/Summaries/"
                  +calc_type+"-Min.csv",index_col=False)

df_tot = pd.concat([df1,df2,df3], ignore_index=True)

df_tot= df_tot.astype({'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint8,
                              "beta":np.uint8,
                              "phi3/pi":np.float64,
                              # "phi3rel/pi":np.float64,
 			      # "FT-J12-ABS":np.float64,
 			      # "FT-J23-ABS":np.float64,
 			      # "FT-J31-ABS":np.float64,
 			       # "FT-Plaq-PHA":np.float64,
 			      # "HE-J12-ABS":np.float64,
 			      # "HE-J23-ABS":np.float64,
 			      # "HE-J31-ABS":np.float64,
 			        "HE-Plaq-PHA":np.float64,
 			       # "FT-LowerT.X":np.float64,
 			       # "FT-LowerT.Y":np.float64,
 			        "HE-LowerT.X":np.float64,
 			        "HE-LowerT.Y":np.float64
                              })

st = time.time()    
df_tot.to_csv("D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/"
              +calc_type+"/"+calc_type+"-Min.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")