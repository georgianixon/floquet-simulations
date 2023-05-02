
import time
import numpy as np
import pandas as pd
from floquet_simulations.flux_functions import *

dataLoc = "D:/Data/Set13-alpha=1,beta=2,omega=9/"
dfO = pd.read_csv(dataLoc+"Summaries/FT-Min.csv",
                          index_col=False)

# dfO = pd.read_csv("D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/FT/FT-Min,phi3=0.csv", 
#                   index_col=False)

radius = 0.05

"""
Pick square in the lower triangle (brackets [ ])
See what sort of phases we can get in that square
"""

dataSave = 'D:/Data/Set13-alpha=1,beta=2,omega=9/HE/'

calc_type = "HE"

rad = 0.05


df = pd.DataFrame(columns = ["CentreX", "CentreY", "Radius", "nPoints",
                             "MaxDelta", "MaxPhaseOpening", "MaxPhaseClosing", 
                             "SecondMaxDelta", "SecondMaxPhaseOpening", "SecondMaxPhaseClosing"])
i = len(df)
for sqCentreX in np.linspace(0,1,101)[:-1]:
    for sqCentreY in np.linspace(0.0, sqCentreX, round((sqCentreX)/0.01 + 1)):

        
        sqCentreX = np.round(sqCentreX, 3)
        sqCentreY = np.round(sqCentreY, 3)
        print(sqCentreX, sqCentreY)
        
        dfP = dfO[((dfO[calc_type+"-LowerT.X"] - sqCentreX)**2 + (dfO[calc_type+"-LowerT.Y"] - sqCentreY)**2 <= radius**2)]
        
        phases = dfP[calc_type+"-Plaq-PHA"].to_numpy()
        n_points = len(phases)

        phases = np.sort(phases)
        maxDelta, secondMaxDelta = MaxDeltas2(phases)
        df.loc[i] = [sqCentreX, sqCentreY, radius, n_points, maxDelta[0], maxDelta[1][0], maxDelta[1][1], 
                     secondMaxDelta[0], secondMaxDelta[1][0], secondMaxDelta[1][1]]
        i +=1

st = time.time()    
df.to_csv(dataSave + "PhasesPlot.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")

dfP = df