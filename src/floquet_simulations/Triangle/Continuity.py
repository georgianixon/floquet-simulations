# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:12:45 2022

@author: Georgia
"""

import numpy as np
import pandas as pd
import time

def ConvertComplex(s):
    """
    For retrieving complex numbers from csv's
    """
    return np.complex128(s.replace('i', 'j').replace('*I', 'j').replace('*^', 'e'))


        
def MinMedMax(a,b,c):
    # a = dfi["FT-J12-ABS"]
    # b = dfi["FT-J23-ABS"]
    # c = dfi["FT-J31-ABS"]
    if min([a,b,c])==a:
        if max([a,b,c])==b:
            return 0
        elif max([a,b,c])==c:
            return 1
    elif min([a,b,c])==b:
        if max([a,b,c])==a:
            return 2
        elif max([a,b,c])==c:
            return 3
    elif min([a,b,c])==c:
        if max([a,b,c])==a:
            return 4
        elif max([a,b,c])==b:
            return 5     

# dataLoc = "D:/Data/Set8-alpha=1,beta=2/Raw/"
dataLoc = "D:/Data/Set12-alpha=1,beta=2,omega=8/"
            
#%%

alpha = 1
beta = 2

dfs_no_raw = []

for A2 in np.linspace(0,30,31):
    
    for A3 in np.linspace(0,30,31):
        # print(str(int(A2)), str(int(A3)))
        
        fileName = ("Raw/TriangleRatios,alpha="+str(int(alpha))
                    +",beta="+str(int(beta))+",A2="+str(int(A2))
                    +",A3="+str(int(A3))+".csv")
        # fileName = "TriangleRatios,alpha=1,beta=2,A2=0,A3=0.csv"# print(fileName)
        
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
                              }
                          )
        print(A2, A3)
        dfi = dfi.drop(columns=["FT-J12","FT-J23","FT-J31",
                                "HE-J12","HE-J23","HE-J31",
                                "HE-O1", "HE-O2","HE-O3",
                                "FT-J12-ABS",
                                "FT-J23-ABS",
                                "FT-J31-ABS",
                                "FT-Plaq-PHA",
                                # "HE-J12-ABS",
                                # "HE-J23-ABS",
                                # "HE-J31-ABS",
                                "HE-Plaq-PHA",
                                "FT-LowerT.X",
                                "FT-LowerT.Y",
                                # "HE-LowerT.X",
                                # "HE-LowerT.Y"
                                ])
          
          #FInd rows with nans
        nanRows = np.where(np.isnan(dfi["HE-J12-ABS"]))[0]
          #drop nan locs
        print("  # nans:", len(nanRows))
        dfi = dfi.drop(nanRows)
          
        # for col in ["FT-J12","FT-J23","FT-J31","HE-J12",
        #               "HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"]:
        #     dfi[col] = ConvertComplex(dfi[col])
          
        # dfi = dfi.drop(columns=["FT-J12","FT-J23","FT-J31","HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"])
        # dfi = dfi[dfi["phi3/pi"]==0]
        
        
        dfi["continuity"] = dfi.apply(lambda x: MinMedMax(x["HE-J12-ABS"], 
                                             x["HE-J23-ABS"],
                                             x["HE-J31-ABS"]), axis=1)
        
        dfs_no_raw.append(dfi)

dfO = pd.concat(dfs_no_raw, ignore_index=True)
            
st = time.time()    
dfO.to_csv(dataLoc + "HE/continuityplots/ContinuityData.csv",
                  index=False, 
                  )
et = time.time() 
print("   save took", np.round(et - st, 1), "s")