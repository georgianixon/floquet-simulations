# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:48:05 2022

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
# dataLoc = "E:/Set4-alpha=1,beta=3,A3e{0,5}/"
# dataLoc = "E:/Set5-alpha=1,beta=3,A3e{6,17}/"
dataLoc = "E:/Set6-alpha=1,beta=3,A3e{18,30}/"
# dataLoc = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/Triangle/Cluster/"
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")





def ListRatiosInLowerTriangle(lst1a,lst1b, lst2a,lst2b, lst3a,lst3b):
    """
    Go thourgh (x1,y1), (x2,y2) (x3,y3) combinations and find the one in the bottom right triangle
    """
    N = len(lst1a)
    lowerTriListA = np.zeros(N)
    lowerTriListB = np.zeros(N)
    
    upperTriListX = np.zeros(N)
    upperTriListY = np.zeros(N)
    
    # counts = np.zeros(N)
    
    for i, (a1, b1, a2, b2, a3, b3) in enumerate(list(zip(lst1a, lst1b, lst2a, lst2b, lst3a, lst3b))):
        # count = 0
        if a1 <=1 and b1 <=1:
            # count +=1
            if b1<=a1:
                lowerTriListA[i] = a1
                lowerTriListB[i] = b1
                
                upperTriListX[i] = b1
                upperTriListY[i] = a1
            else:
                lowerTriListA[i] = b1
                lowerTriListB[i] = a1
                
                upperTriListX[i] = a1
                upperTriListY[i] = b1
        elif a2 <=1 and b2 <=1:
            # count +=1
            if b2<=a2:
                lowerTriListA[i] = a2
                lowerTriListB[i] = b2
                
                upperTriListX[i] = b2
                upperTriListY[i] = a2
            else:
                lowerTriListA[i] = b2
                lowerTriListB[i] = a2
                
                upperTriListX[i] = a2
                upperTriListY[i] = b2
        elif a3 <=1 and b3 <=1:
            # count+=1
            if b3<=a3:
                lowerTriListA[i] = a3
                lowerTriListB[i] = b3
                
                upperTriListX[i] = b3
                upperTriListY[i] = a3
            else:
                lowerTriListA[i] = b3
                lowerTriListB[i] = a3
                
                upperTriListX[i] = a3
                upperTriListY[i] = b3
                
        # counts[i] = count
        # else:
        #     print(i)
        #     raise ValueError
    return lowerTriListA, lowerTriListB, upperTriListX, upperTriListY


#%%
alpha = 1
beta = 3


dfs_no_raw = []

# folders = [ "Set5"]
# for f in folders:
for A2 in np.linspace(0,30,31):
    
    for A3 in np.linspace(18,30,13):
        # print(str(int(A2)), str(int(A3)))
        
        fileName = "TriangleRatios,alpha="+str(int(alpha))+",beta="+str(int(beta))+",A2="+str(int(A2))+",A3="+str(int(A3))+".csv"
        # fileName = "TriangleRatios,alpha=1,beta=2,A2=0,A3=0.csv"# print(fileName)
        try:
            dfO = pd.read_csv(dataLoc+fileName,
                          index_col=False   ,
                          dtype = {'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint32,
                              "beta":np.uint32,
                              "phi3/pi":np.float64
                              } 
          #                 ,
          #                 converters={"FT-J12":ConvertComplex,
 			      # "FT-J23":ConvertComplex,
 			      # "FT-J31":ConvertComplex,
 			      # "HE-J12":ConvertComplex,
 			      # "HE-J23":ConvertComplex,
 			      # "HE-J31":ConvertComplex,
          #         "HE-O1":ConvertComplex,
 			      # "HE-O2":ConvertComplex,
 			      # "HE-O3":ConvertComplex,}
                    )
            
            for col in ["FT-", "HE-"]:
                for v in ["J12", "J31", "J23"]:
                    # print(col, v)
                    dfO[col+v] = ConvertComplex(dfO[col+v])
            
            print(A2, A3)
            
            
            dfO["FT-J12-ABS"] = np.abs(dfO["FT-J12"])
            dfO["FT-J23-ABS"] = np.abs(dfO["FT-J23"])
            dfO["FT-J31-ABS"] = np.abs(dfO["FT-J31"])
            
            dfO["HE-J12-ABS"] = np.abs(dfO["HE-J12"])
            dfO["HE-J23-ABS"] = np.abs(dfO["HE-J23"])
            dfO["HE-J31-ABS"] = np.abs(dfO["HE-J31"])
            
            dfO["FT-J12/J23"] = dfO["FT-J12-ABS"] / dfO["FT-J23-ABS"]
            dfO["FT-J31/J23"] = dfO["FT-J31-ABS"] / dfO["FT-J23-ABS"]
            dfO["FT-J31/J12"] = dfO["FT-J31-ABS"] / dfO["FT-J12-ABS"]
            dfO["FT-J23/J12"] = dfO["FT-J23-ABS"] / dfO["FT-J12-ABS"]
            dfO["FT-J23/J31"] = dfO["FT-J23-ABS"] / dfO["FT-J31-ABS"]
            dfO["FT-J12/J31"] = dfO["FT-J12-ABS"] / dfO["FT-J31-ABS"]
            
            dfO["HE-J12/J23"] = dfO["HE-J12-ABS"] / dfO["HE-J23-ABS"]
            dfO["HE-J31/J23"] = dfO["HE-J31-ABS"] / dfO["HE-J23-ABS"]
            dfO["HE-J31/J12"] = dfO["HE-J31-ABS"] / dfO["HE-J12-ABS"]
            dfO["HE-J23/J12"] = dfO["HE-J23-ABS"] / dfO["HE-J12-ABS"]
            dfO["HE-J23/J31"] = dfO["HE-J23-ABS"] / dfO["HE-J31-ABS"]
            dfO["HE-J12/J31"] = dfO["HE-J12-ABS"] / dfO["HE-J31-ABS"]
            
            """
            get point on lower triangle
            """
            for col in ["FT-", "HE-"]:
                x = dfO[col+"J12/J23"].to_numpy()
                y = dfO[col+"J31/J23"].to_numpy()
                t = dfO[col+"J23/J12"].to_numpy()
                d = dfO[col+"J31/J12"].to_numpy() 
                s = dfO[col+"J23/J31"].to_numpy() 
                p = dfO[col+"J12/J31"].to_numpy() 
            
                lowerTriListA, lowerTriListB, upperTriListX, upperTriListY = ListRatiosInLowerTriangle(x, y, t, d, s, p)
            
                dfO[col+"LowerT.X"] = lowerTriListA
                dfO[col+"LowerT.Y"] = lowerTriListB
                # dfO[col+"UpperT.X"] = upperTriListX
                # dfO[col+"UpperT.Y"] = upperTriListY
                    
            
                dfO[col+"Plaq-PHA"]= np.angle(dfO[col+"J31"])

            dfO = dfO.drop(columns=["FT-J12","FT-J23","FT-J31","HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"])
            for col in ["FT-", "HE-"]:
                for v1 in ["J12", "J23", "J31"]:
                    for v2 in ["J12", "J23", "J31"]:
                        if v1 != v2:
                            dfO = dfO.drop(columns=[col+v1+"/"+v2])
                        
                        
            dfO = dfO.astype({'A2': np.float64,
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
            
            #FInd rows with nans
            nanRows = np.where(np.isnan(ConvertComplex(dfO["FT-J12-ABS"])))[0]
            #drop nan locs
            print("  # nans:", len(nanRows))
            dfO = dfO.drop(nanRows)
    
            dfs_no_raw.append(dfO)
            
        except:
            p = 0
            print("Oh no ", A2, A3)
            


bigData = pd.concat(dfs_no_raw, ignore_index=True)

bigData = bigData.reindex(columns=['A2', 'A3', 'omega0', 'alpha', 'beta', 'phi3/pi', 'FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'FT-Plaq-PHA', 'HE-J12-ABS', 'HE-J23-ABS', 'HE-J31-ABS', 'HE-Plaq-PHA',
       'FT-LowerT.X', 'FT-LowerT.Y', 'HE-LowerT.X',
       'HE-LowerT.Y'])


st = time.time()   

save_set = "Set5" 

bigData.to_csv(dataLoc +  "Summary.csv",
                  index=False, 
                  )
et = time.time()
print("   save took", np.round(et - st, 1), "s")





# save phases data only
dfPha = bigData.drop(columns=[ 'FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'HE-J12-ABS', 'HE-J23-ABS',
       'HE-J31-ABS',  'FT-LowerT.X', 'FT-LowerT.Y',
       'HE-LowerT.X', 'HE-LowerT.Y'] )

st = time.time()    
dfPha.to_csv(dataLoc + "Phases.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


# save lower triangle data only
dfLowerT = bigData.drop(columns=['FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'FT-Plaq-PHA', 'HE-J12-ABS', 'HE-J23-ABS',
       'HE-J31-ABS', 'HE-Plaq-PHA' ] )
st = time.time()    
dfLowerT.to_csv(dataLoc + "LowerTriangle.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")



# save HT data only
dfHE = bigData.drop(columns=['FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS', 'FT-Plaq-PHA', 'FT-LowerT.X', 'FT-LowerT.Y' ] )
st = time.time()    
dfHE.to_csv(dataLoc + "HE.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


# save FT data only
dfFT = bigData.drop(columns=['HE-J12-ABS',
       'HE-J23-ABS', 'HE-J31-ABS', 'HE-Plaq-PHA', 'HE-LowerT.X', 'HE-LowerT.Y' ] )
st = time.time()    
dfFT.to_csv(dataLoc + "FT.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")

