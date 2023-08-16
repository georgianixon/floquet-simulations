from scipy.special import jv
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import pandas as pd
import matplotlib as mpl
from floquet_simulations.flux_functions import *
from pathlib import Path
from floquet_simulations.hamiltonians import ConvertComplex

omega0 = 8

data_dirs = [Path(__file__).absolute().parent.parent/"paper_data"/f"Heff_omega={omega0},alpha={2},beta={3}.csv",
             Path(__file__).absolute().parent.parent/"paper_data"/f"Heff_omega={omega0},alpha={1},beta={2}.csv"]

gaps_dir = Path(__file__).absolute().parent.parent/"paper_data"/f"neighbourhood_continuous_gaps_omega={omega0},(alpha,beta)=[(1,2),(2,3)],Amax=70.csv"


df = pd.DataFrame(columns = ["A2","A3","omega0","alpha","beta","phi3/pi","FT-J12","FT-J23","FT-J31","FT-LowerT.X","FT-LowerT.Y","xi"])
df = df.astype({
        'A2': np.float32,
                    'A3': np.float64,
                    'omega0': np.float64,
                    "alpha":np.uint32,
                        "beta":np.uint32,
                        "phi3/pi":np.float64,
                    "FT-J12":np.complex128,
                    "FT-J23":np.complex128,
                    "FT-J31":np.complex128,
                    #  "HE-J12":np.complex128,
                    #  "HE-J23":np.complex128,
                    #  "HE-J31":np.complex128,
                    #  "HE-O1":np.complex128,
                    #  "HE-O2":np.complex128,
                    #  "HE-O3":np.complex128
                    "FT-LowerT.X":np.float64,
                    "FT-LowerT.Y":np.float64,
                    "xi":np.float64
                    })
for dir in data_dirs:

    df1 = pd.read_csv(dir,
                    index_col=False,
                    converters={"FT-J12": ConvertComplex,
                                    "FT-J23": ConvertComplex,
                                    "FT-J31": ConvertComplex,
                    #                   # "HE-J12": ConvertComplex,
                    #                   # "HE-J23": ConvertComplex,
                    #                   # "HE-J31": ConvertComplex,
                    #                   # "HE-O1": ConvertComplex,
                    #                   # "HE-O2": ConvertComplex,
                    #                   # "HE-O3": ConvertComplex
                    }
                    )

    df1["phi3/pi"]= np.round(df1["phi3/pi"], 2)
    xi_fixed =  np.angle(df1["FT-J23"].to_numpy())+np.angle(df1["FT-J12"].to_numpy())+np.angle(df1["FT-J31"].to_numpy())
    xi_fixed = (xi_fixed + 2*pi)%(2*pi)
    xi_fixed = np.where(xi_fixed>pi, xi_fixed-2*pi, xi_fixed)
    df1["xi"] = xi_fixed
    df = pd.concat([df, df1], ignore_index = True)

print("concat df")

df = df.groupby(by=[ "A2", "A3", "omega0", "alpha", "beta", "phi3/pi" ]).agg({'FT-J12':"mean", 
                                            'FT-J23':"mean", 
                                            'FT-J31':"mean", 
                                            "FT-LowerT.X":"mean",
                                            "FT-LowerT.Y":"mean",
                                                    "xi":"mean"
                        }).reset_index()

df = df.astype({
        'A2': np.float32,
                    'A3': np.float64,
                    'omega0': np.float64,
                    "alpha":np.uint32,
                        "beta":np.uint32,
                        "phi3/pi":np.float64,
                    "FT-J12":np.complex128,
                    "FT-J23":np.complex128,
                    "FT-J31":np.complex128,
                    #  "HE-J12":np.complex128,
                    #  "HE-J23":np.complex128,
                    #  "HE-J31":np.complex128,
                    #  "HE-O1":np.complex128,
                    #  "HE-O2":np.complex128,
                    #  "HE-O3":np.complex128
                    "FT-LowerT.X":np.float64,
                    "FT-LowerT.Y":np.float64,
                    "xi":np.float64
                    })



df_gaps = pd.DataFrame(columns = ["DataType","CentreX","CentreY","Radius", "FirstMaxDelta","FirstMaxPhaseOpening","FirstMaxPhaseClosing"
                            #  ,"SecondMaxDelta","SecondMaxPhaseOpening","SecondMaxPhaseClosing"
                            #  ,"ThirdMaxDelta","ThirdMaxPhaseOpening","ThirdMaxPhaseClosing",
                            #  "FourthMaxDelta","FourthMaxPhaseOpening","FourthMaxPhaseClosing"
                             ])

df_gaps.to_csv(gaps_dir, index=False)

radius = 0.05

j = len(df_gaps)
for sqCentreX in np.linspace(0,1,101)[:-1]:
    for sqCentreY in np.linspace(0.0, sqCentreX, round((sqCentreX)/0.01 + 1)):

        
        sqCentreX = np.round(sqCentreX, 3)
        sqCentreY = np.round(sqCentreY, 3)
        print(sqCentreX, sqCentreY)

        dfP = df[((df["FT-LowerT.X"] - sqCentreX)**2 + (df["FT-LowerT.Y"] - sqCentreY)**2 <= radius**2)
        #  &(df.A2 >=60)& (df.A3 <= 60)
         ]
        dfP.reset_index(drop=True, inplace=True)

        # find continuous flux lines with phi3
        last_phi3 = dfP.iloc[0]["phi3/pi"] #set first phi3 val

        # phi3_biglist = [[last_phi3]]
        xi_biglist=[[dfP.iloc[0]["xi"]]]
        # a3_biglist = [dfP.iloc[0]["A3"]]
        # a2_biglist = [dfP.iloc[0]["A2"]]
        # last_sign = np.sign(dfP.iloc[0]["xi"])
        lst_num=0
        for _, row in dfP[1:].iterrows():
            xi = np.real(row["xi"])
            phi3= np.real(row["phi3/pi"])
            # a2 = np.real(row.A2)
            # a3 = np.real(row.A3)
            
            if np.round(phi3,2)==np.round(last_phi3+0.02, 2):

                xi_biglist[lst_num].append(xi)
                # phi3_biglist[lst_num].append(phi3)

            else:
                xi_biglist.append([np.real(xi)])
                # phi3_biglist.append([phi3])
                # a2_biglist.append(a2)
                # a3_biglist.append(a3)

                lst_num+=1
            
            last_phi3= phi3
            
        for i in range(len(xi_biglist)):
            xi_biglist[i][0]=np.real(xi_biglist[i][0])
            # a2_biglist[i]=np.real(a2_biglist[i])
            # a3_biglist[i]=np.real(a3_biglist[i])
            # phi3_biglist[i][0]=np.real(phi3_biglist[i][0])

        # get min and max points of the line
        xi_minmaxs= []
        for xi_list in xi_biglist:
            if len(xi_list) >1:
                xi_minmaxs.append((np.round(np.min(np.abs(xi_list)), 5), np.round(np.max(np.abs(xi_list)), 5)))
        xi_minmaxs = sorted(xi_minmaxs)

        # find gaps in the coverage
        gaps = []
        cover_bottom = xi_minmaxs[0][0]
        cover_top = xi_minmaxs[0][1]
        for cover_line in xi_minmaxs[1:]:
            if cover_line[0] == cover_bottom:
                # extend cover_top
                cover_top = cover_line[1]
            else:
                # new, higher bottom
                if cover_line[0] <= cover_top:
                    #great, no gap
                    if cover_line[1] > cover_top:
                        #extend cover_top
                        cover_top = cover_line[1]
                    #if cover_line[1] < cover_top:
                        #nothing to be gained
                else:
                    # we have a gap
                    gaps.append((cover_top, cover_line[0]))
                    # change cover bottom and top
                    cover_bottom = cover_line[0]
                    cover_top = cover_line[1]
        if gaps:
            max_gap = sorted([(gap_range[1] - gap_range[0], gap_range[0], gap_range[1]) for gap_range in gaps], reverse=True)
            
            df_gaps.loc[j] = ["FT", sqCentreX, sqCentreY, radius,
                        max_gap[0][0], max_gap[0][1], max_gap[0][2],
                        #  thirdMaxDelta[0], thirdMaxDelta[1][0], thirdMaxDelta[1][1], 
                        #  fourthMaxDelta[0], fourthMaxDelta[1][0], fourthMaxDelta[1][1]
                        ]
        else:
            df_gaps.loc[j] = ["FT", sqCentreX, sqCentreY, radius,
                                    0, np.nan, np.nan,
                                    #  thirdMaxDelta[0], thirdMaxDelta[1][0], thirdMaxDelta[1][1], 
                                    #  fourthMaxDelta[0], fourthMaxDelta[1][0], fourthMaxDelta[1][1]
                                    ]
        j +=1

df_gaps.to_csv(gaps_dir, index=False)
