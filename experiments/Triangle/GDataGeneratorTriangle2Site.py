# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 15:13:19 2021

@author: Georgia Nixon
"""


place = "Georgia"

from numpy.linalg import eig
from numpy import  pi, log, exp, sin
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd 
import time
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/floquet-simulations/src')
from hamiltonians import  CreateHFGeneral, SolveSchrodinger, ConvertComplex, Cosine, Ramp, RampHalf, Blip
dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"

def filter_duplicates(x):
    """
    input dataframe, df.x, eg. df.localisation
    output value 
    """
    xx = []
    # get only values
    for i in x:  #for the values in the df x
        if not np.isnan(i):
            xx.append(i)    
    if len(xx)==0:
        return np.nan
    else:
        xxx = [np.round(i, 2) for i in xx]
        if len(set(xxx))==1:
            return np.mean(xx)
        else:
            return np.nan
        

dfname = "analysis-G-Triangle-2Site.csv"
dfname = "analysis-G-Triangle-2Site-RemoveGauge.csv"

def RemoveWannierGauge(matrix, p):
    phase = np.angle(N[p-1,p])
    for i in [p-1, p+1]:
        matrix[i, p] = np.exp(-1j*phase)*matrix[i,p]
        matrix[p, i] = np.exp(1j*phase)*matrix[p,i]
    return matrix



# df = pd.DataFrame(columns=["form", "func1","func2","rtol","N", 
#                             "centre1","centre2",
#                             "a1", "a2", 
#                             "omega1", "omega2", 
#                             "phi1", "phi2", 
#                             "onsite1","onsite2",
#                             "O-1",
#                             "O-2",
#                             "O-3",
#                             "N1-1",
#                             "N1-2",
#                             "N1-3"
#                           ])
    
# df.to_csv(dataLoc+dfname,
#                     index=False, 
#                     columns=["form", "func1","func2","rtol","N", 
#                             "centre1","centre2",
#                             "a1", "a2", 
#                             "omega1", "omega2", 
#                             "phi1", "phi2", 
#                             "onsite1","onsite2",
#                             "O-1",
#                             "O-2",
#                             "O-3",
#                             "N1-1",
#                             "N1-2",
#                             "N1-3"]
#                     )

#%%
# df_dtype_dict = {'form':str,'func':str, "rtol":np.float64, 'N':int, "centre":int,
#                  'a':np.float64, 
#                  'omega':np.float64, 
#                  'phi':np.float64,
#                  "onsite":np.float64,
#                  "O-3":np.float64,
#                 "O-2":np.float64,
#                 "O-1":np.float64,
#                 "O":np.float64,
#                 "O+1":np.float64,
#                 "O+2":np.float64,
#                 "O+3":np.float64,
#                 "N1-3":np.float64,
#                 "N1-2":np.float64,
#                 "N1-1":np.float64,
#                 "N1+1":np.float64,
#                 "N1+2":np.float64,
#                 "N1+3":np.float64,
#                 "N2-2":np.float64,
#                 "N2-1":np.float64,
#                 "N2":np.float64,
#                 "N2+1":np.float64,
#                 "N2+2":np.float64,
#                 "N3-2":np.float64,
#                 "N3-1":np.float64,
#                 "N3+1":np.float64,
#                 "N3+2":np.float64,
#                 "N4-1":np.float64,
#                 "N4":np.float64,
#                 "N4+1":np.float64,
#                 "N5-1":np.float64,
#                 "N5+1":np.float64,
#                 "N6":np.float64}

df = pd.read_csv(dataLoc+dfname, 
                 index_col=False, 
                 converters={
                            "O-1": ConvertComplex,
                            "O-2": ConvertComplex,
                            "O-3": ConvertComplex,
                            "N1-1": ConvertComplex,
                            "N1-2": ConvertComplex,
                            "N1-3": ConvertComplex
                            })



#%%

# need tp dp 1e-6 phi = 0
N = 3; 
centre1=1; centre2=2
centres=[centre1,centre2];
form="Tri"#'SS-p' 


rtol = 1e-11
phis = [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
omegas = np.linspace(3.1, 20, int((20-3.1)*10+1), endpoint=True)
# omegas = np.linspace(20.1, 200, 200*10-200)

onsite1 = 0; onsite2 = 0
funcs = [Blip, Blip]
funcname1 = "Blip"; funcname2="Blip"

circleBoundary = 1

# UT, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)



    
for a in [35]:
    
    
    for phi1 in phis:
        for phi2 in phis:
            print('a=',a,'  phi1=',phi1,'  phi2=',phi2)
            df1 = pd.DataFrame(columns=["form", "func1","func2","rtol","N", 
                                        "centre1","centre2",
                                        "a1", "a2", 
                                        "omega1", "omega2", 
                                        "phi1", "phi2", 
                                        "onsite1","onsite2",
                                        "O-1",
                                        "O-2",
                                        "O-3",
                                         "N1-1",
                                         "N1-2",
                                         "N1-3"])
            for i, omega1 in enumerate(omegas):
                
                start = time.time()
                
                omega1 = round(omega1, 1)
                print(omega1)
                
                T = 2*pi/omega1
                omega2 = 2*omega1
                # elif form =="DS-p" or form == "SSDF-p":
                #     omega2 = omegaMultiplier*omega1
                #     aInput = [a1,a2]
                #     omegaInput = [omega1,omega2]
                #     phiInput = [phi, phi+phiOffset]
        
                # calculate effective Hamiltonian 
                paramss = [[a, omega1, phi1, onsite1], [a, omega2, phi2, onsite2]]
                UT, HF = CreateHFGeneral(N, centres, funcs, paramss, T, circleBoundary)
        
                
                # log matrix elements
                Om1 = HF[0][0]
                Om2 = HF[1][1]
                Om3 = HF[2][2]
                
                N1m1 = HF[0][1]
                N1m2 = HF[1][2]
                N1m3 = HF[0][2]
              
                
                
                
                print('   ',time.time()-start, 's')
                
                df1.loc[i] = [form, 
                              funcname1,
                              funcname2,
                              rtol,
                              N,
                              centre1,
                              centre2,
                              a,
                              a,
                              omega1,
                              omega2,
                              phi1,
                              phi2,
                              onsite1,
                              onsite2,
                              Om1,
                                Om2,
                                Om3,
                                
                                N1m1, 
                                N1m2,
                                N1m3, 
                               ]
        
        
            df = df.append(df1, ignore_index=True, sort=False)
            # df= df.astype(dtype=df_dtype_dict)
            
        #        print('  grouping..')
        #        df = df.groupby(by=['form', 'rtol', 'a', 'omega', 'phi', 
        #                         'N'], dropna=False).agg({
        #                                'hopping':filter_duplicates,
        #                                'onsite':filter_duplicates,
        #                                'next onsite':filter_duplicates,
        #                                'NNN':filter_duplicates,
        #                                'NNN overtop':filter_duplicates
        #                                }).reset_index()
            
            print('   saving..')
            df.to_csv(dataLoc+dfname,
                      index=False, 
                      columns=["form", "func1","func2","rtol","N", 
                                        "centre1","centre2",
                                        "a1", "a2", 
                                        "omega1", "omega2", 
                                        "phi1", "phi2", 
                                        "onsite1","onsite2",
                              "O-1",
                                "O-2",
                                "O-3",
                                 "N1-1",
                                 "N1-2",
                                 "N1-3"])