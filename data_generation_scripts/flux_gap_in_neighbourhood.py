
import time
import numpy as np
import pandas as pd
from floquet_simulations.flux_functions import *
from pathlib import Path
from floquet_simulations.hamiltonians import ConvertComplex
from numpy import pi
# dataLoc = "D:/Data/Set13-alpha=1,beta=2,omega=9/"
omega0=8; alpha=1;beta=2
df_dir = Path(__file__).absolute().parent.parent/"paper_data"/f"Heff_omega={omega0},alpha={alpha},beta={beta}.csv"
dfO = pd.read_csv(df_dir,
                      index_col=False, 
                        converters={"FT-J12": ConvertComplex,
                                  "FT-J23": ConvertComplex,
                                  "FT-J31": ConvertComplex,
                                  # "HE-J12": ConvertComplex,
                                  # "HE-J23": ConvertComplex,
                                  # "HE-J31": ConvertComplex,
                                  # "HE-O1": ConvertComplex,
                                  # "HE-O2": ConvertComplex,
                                  # "HE-O3": ConvertComplex
                                    }
                       )

xi_fixed =  np.angle(dfO["FT-J23"].to_numpy())+np.angle(dfO["FT-J12"].to_numpy())+np.angle(dfO["FT-J31"].to_numpy())
xi_fixed = (xi_fixed + 2*pi)%(2*pi)
xi_fixed = np.where(xi_fixed>pi, xi_fixed-2*pi, xi_fixed)
dfO["xi"] = xi_fixed

# 

# dfO = pd.read_csv(dataLoc+"Summaries/FT-Min.csv",
#                           index_col=False)

# dfO = pd.read_csv("D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/FT/FT-Min,phi3=0.csv", 
#                   index_col=False)

radius = 0.05

"""
Pick square in the lower triangle (brackets [ ])
See what sort of phases we can get in that square
"""

# dataSave = 'D:/Data/Set13-alpha=1,beta=2,omega=9/HE/'
save_dir = Path(__file__).absolute().parent.parent/"paper_data"/f"neighbourhood_gaps_omega={omega0},alpha={alpha},beta={beta},Amax=70.csv"

calc_type="FT"
rad = 0.05


df = pd.DataFrame(columns = ["DataType","CentreX","CentreY","Radius","nPoints",
                             "FirstMaxDelta","FirstMaxPhaseOpening","FirstMaxPhaseClosing",
                             "SecondMaxDelta","SecondMaxPhaseOpening","SecondMaxPhaseClosing"
                            #  ,"ThirdMaxDelta","ThirdMaxPhaseOpening","ThirdMaxPhaseClosing",
                            #  "FourthMaxDelta","FourthMaxPhaseOpening","FourthMaxPhaseClosing"
                             ])

st = time.time()    
df.to_csv(save_dir, index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")

i = len(df)
for sqCentreX in np.linspace(0,1,101)[:-1]:
    for sqCentreY in np.linspace(0.0, sqCentreX, round((sqCentreX)/0.01 + 1)):

        
        sqCentreX = np.round(sqCentreX, 3)
        sqCentreY = np.round(sqCentreY, 3)
        print(sqCentreX, sqCentreY)
        
        dfP = dfO[((dfO[calc_type+"-LowerT.X"] - sqCentreX)**2 + (dfO[calc_type+"-LowerT.Y"] - sqCentreY)**2 <= radius**2)]
        
        phases = dfP["xi"].to_numpy()
        n_points = len(phases)

        phases = np.sort(phases)
        firstMaxDelta, secondMaxDelta = MaxDeltas2(phases, num_gaps=2)
        df.loc[i] = [calc_type, sqCentreX, sqCentreY, radius, n_points, 
                     firstMaxDelta[0], firstMaxDelta[1][0], firstMaxDelta[1][1], 
                     secondMaxDelta[0], secondMaxDelta[1][0], secondMaxDelta[1][1],
                    #  thirdMaxDelta[0], thirdMaxDelta[1][0], thirdMaxDelta[1][1], 
                    #  fourthMaxDelta[0], fourthMaxDelta[1][0], fourthMaxDelta[1][1]
                     ]
        i +=1

st = time.time()    
df.to_csv(save_dir, index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")