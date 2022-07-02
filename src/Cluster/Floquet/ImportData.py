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

# dataLoc = "/Users/"+place+"/Code/MBQD/floquet-simulations/src/Cluster/Floquet/Data/"
dataLoc = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/Triangle/Cluster/"
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")

def GetNewColumns(dfO):
    dfO["FT-J12-ABS"] = np.abs(dfO["FT-J12"])
    dfO["FT-J23-ABS"] = np.abs(dfO["FT-J23"])
    dfO["FT-J31-ABS"] = np.abs(dfO["FT-J31"])
    
    #negative to account for the fact that matrix element is negative
    dfO["FT-J12-PHA"] = np.angle(-dfO["FT-J12"])
    dfO["FT-J23-PHA"] = np.angle(-dfO["FT-J23"])
    dfO["FT-J31-PHA"] = np.angle(-dfO["FT-J31"])
    
    
    dfO["HE-J12-ABS"] = np.abs(dfO["HE-J12"])
    dfO["HE-J23-ABS"] = np.abs(dfO["HE-J23"])
    dfO["HE-J31-ABS"] = np.abs(dfO["HE-J31"])
    
    #negative to account for the fact that matrix element is negative
    dfO["HE-J12-PHA"] = np.angle(-dfO["HE-J12"])
    dfO["HE-J23-PHA"] = np.angle(-dfO["HE-J23"])
    dfO["HE-J31-PHA"] = np.angle(-dfO["HE-J31"])
    
    
    dfO["FT-J12/J23-ABS"] = dfO["FT-J12-ABS"] / dfO["FT-J23-ABS"]
    dfO["FT-J31/J23-ABS"] = dfO["FT-J31-ABS"] / dfO["FT-J23-ABS"]
    dfO["FT-J31/J12-ABS"] = dfO["FT-J31-ABS"] / dfO["FT-J12-ABS"]
    dfO["FT-J23/J12-ABS"] = dfO["FT-J23-ABS"] / dfO["FT-J12-ABS"]
    dfO["FT-J23/J31-ABS"] = dfO["FT-J23-ABS"] / dfO["FT-J31-ABS"]
    dfO["FT-J12/J31-ABS"] = dfO["FT-J12-ABS"] / dfO["FT-J31-ABS"]
    
    
    dfO["HE-J12/J23-ABS"] = dfO["HE-J12-ABS"] / dfO["HE-J23-ABS"]
    dfO["HE-J31/J23-ABS"] = dfO["HE-J31-ABS"] / dfO["HE-J23-ABS"]
    dfO["HE-J31/J12-ABS"] = dfO["HE-J31-ABS"] / dfO["HE-J12-ABS"]
    dfO["HE-J23/J12-ABS"] = dfO["HE-J23-ABS"] / dfO["HE-J12-ABS"]
    dfO["HE-J23/J31-ABS"] = dfO["HE-J23-ABS"] / dfO["HE-J31-ABS"]
    dfO["HE-J12/J31-ABS"] = dfO["HE-J12-ABS"] / dfO["HE-J31-ABS"]
    
    
    # """
    # get point on lower triangle
    # """
    
    lowerTriListA, lowerTriListB, upperTriListX, upperTriListY = ListRatiosInLowerTriangle(dfO["FT-J12/J23-ABS"].to_numpy(),
                                                                                           dfO["FT-J31/J23-ABS"].to_numpy(), 
                                                                                           dfO["FT-J23/J12-ABS"].to_numpy(), 
                                                                                           dfO["FT-J31/J12-ABS"].to_numpy(),
                                                                                           dfO["FT-J23/J31-ABS"].to_numpy(),
                                                                                           dfO["FT-J12/J31-ABS"].to_numpy())
    dfO["FT-LowerT.X"] = lowerTriListA
    dfO["FT-LowerT.Y"] = lowerTriListB
    # dfO["FT-UpperT.X"] = upperTriListX
    # dfO["FT-UpperT.Y"] = upperTriListY
    
    lowerTriListA, lowerTriListB, upperTriListX, upperTriListY = ListRatiosInLowerTriangle(dfO["HE-J12/J23-ABS"].to_numpy(),
                                                                                           dfO["HE-J31/J23-ABS"].to_numpy(), 
                                                                                           dfO["HE-J23/J12-ABS"].to_numpy(), 
                                                                                           dfO["HE-J31/J12-ABS"].to_numpy(),
                                                                                           dfO["HE-J23/J31-ABS"].to_numpy(),
                                                                                           dfO["HE-J12/J31-ABS"].to_numpy())
    dfO["HE-LowerT.X"] = lowerTriListA
    dfO["HE-LowerT.Y"] = lowerTriListB
    # dfO["HE-UpperT.X"] = upperTriListX
    # dfO["HE-UpperT.Y"] = upperTriListY
    
    
    dfO = dfO.drop(columns=["HE-J12/J23-ABS","HE-J31/J23-ABS","HE-J12/J31-ABS","HE-J23/J31-ABS",
                            "HE-J23/J12-ABS","HE-J31/J12-ABS"])
    dfO = dfO.drop(columns=["FT-J12/J23-ABS","FT-J31/J23-ABS","FT-J12/J31-ABS","FT-J23/J31-ABS",
                            "FT-J23/J12-ABS","FT-J31/J12-ABS"])
    # "FT-J12-ABS","FT-J23-ABS","FT-J31-ABS","HE-J12-ABS","HE-J23-ABS","HE-J31-ABS",
    dfO = dfO.drop(columns=["FT-J12-PHA","FT-J23-PHA",
                            "HE-J12-PHA","HE-J23-PHA"])
    # dfO=dfO.drop(columns=["FT-UpperT.X", "FT-UpperT.Y", "HE-UpperT.X", "HE-UpperT.Y"])
    return dfO


bigData = pd.DataFrame({'A2': [], 
                        'A3': [], 
                        'omega0': [], 
                        'alpha': [], 
                        'beta': [],
                        'phi3/pi': [],                
                        "FT-J12": [],
                        "FT-J23": [],
                        "FT-J31": [],
                        "HE-J12": [],
                        "HE-J23": [],
                        "HE-J31": [],
                        "HE-O1": [],
                        "HE-O2": [],
                        "HE-O3":[],
                        "FT-J12-ABS":[],
                        "FT-J23-ABS":[],
                        "FT-J31-ABS":[],
                        "FT-J31-PHA":[],
                        "HE-J12-ABS":[],
                        "HE-J23-ABS":[],
                        "HE-J31-ABS":[],
                        "HE-J31-PHA":[],
                        "FT-LowerT.X":[],
                        "FT-LowerT.Y":[],
                        "HE-LowerT.X":[],
                        "HE-LowerT.Y":[]
                        })



alpha = 1
beta = 3


folders = [ "Set5"]
for f in folders:
    for A2 in np.linspace(0,30,31):
        
        for A3 in np.linspace(0,30,31):
            # print(str(int(A2)), str(int(A3)))
            fileName = "TriangleRatios,alpha="+str(int(alpha))+",beta="+str(int(beta))+",A2="+str(int(A2))+",A3="+str(int(A3))+".csv"
            # print(fileName)
            try:
                dfi = pd.read_csv(dataLoc+f + "/"+fileName,
                              index_col=False, 
                                converters={"FT-J12": ConvertComplex,
                                          "FT-J23": ConvertComplex,
                                          "FT-J31": ConvertComplex,
                                          "HE-J12": ConvertComplex,
                                          "HE-J23": ConvertComplex,
                                          "HE-J31": ConvertComplex,
                                          "HE-O1": ConvertComplex,
                                          "HE-O2": ConvertComplex,
                                          "HE-O3": ConvertComplex
                                            }
                              )
                dfi = GetNewColumns(dfi)
                print(A2, A3)
                bigData = pd.concat([bigData, dfi], ignore_index=True)
            except:
                p = 0
    
# bigData1 = bigData[bigData.A3<12]
    

bigData.to_csv(dataLoc + "Set5/" + "Set5SummaryPartial.csv",
                  index=False, 
                   columns=['A2', 'A3', 'omega0', 'alpha', 'beta', 'phi3/pi', 
                            # 'FT-J12', 'FT-J23',
       # 'FT-J31', 'HE-J12', 'HE-J23', 'HE-J31',
       'HE-O1', 'HE-O2', 'HE-O3',
       'FT-J12-ABS', 'FT-J23-ABS', 'FT-J31-ABS', 'FT-J31-PHA', 'HE-J12-ABS',
       'HE-J23-ABS', 'HE-J31-ABS', 'HE-J31-PHA', 'FT-LowerT.X', 'FT-LowerT.Y',
       'HE-LowerT.X', 'HE-LowerT.Y']
                  )


bigData.to_csv(dataLoc + "Set5Summary.csv",
                  index=False, 
                  # columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31"]
                  )
#%%




import platform
platform.architecture()
