# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:56:45 2022

@author: Georgia
"""
from scipy.special import jv
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from numpy import pi, exp, sin, cos
from math import gcd
import pandas as pd
place = "Georgia"
import matplotlib as mpl
import seaborn as sns
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
from hamiltonians import CreateHFGeneral, CreateHFGeneralLoopA
from hamiltonians import Cosine,  RemoveWannierGauge
from hamiltonians import Cosine, ConvertComplex
import time

dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/OldStuff/"


dfnamePast = "TriangleRatios-phasedata-v14.csv"
dfname = "TriangleRatios-phasedata-v14-v2.csv"



dfN = pd.read_csv(dataLoc+dfnamePast, 
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

# dfN = pd.DataFrame(columns=["A2", "A3", "omega0", "alpha", "beta",  "phi3/pi",
#                             "FT-J12", "FT-J23", "FT-J31", 
#                             "HE-J12", "HE-J23", "HE-J31",
#                             "HE-O1", "HE-O2", "HE-O3"  ])

dfN = dfN.astype({'A2': np.float64,
                          'A3': np.float64,
                          'omega0': np.float64,
                          "alpha":np.uint8,
                          "beta":np.uint8,
                          "phi3/pi":np.float64,
                          "FT-J12":np.complex128,
                          "FT-J23":np.complex128,
                          "FT-J31":np.complex128,
                          "HE-J12":np.complex128,
                          "HE-J23":np.complex128,
                          "HE-J31":np.complex128,
                          "HE-O1":np.complex128,
                          "HE-O2":np.complex128,
                          "HE-O3":np.complex128
                          })

# df18.drop(df18[(df18.A2 == 17)&(df18.A3==11)].index, inplace=True)

#%%

# A2s = np.linspace(0,4,5)
A2s = [4]
A3s = np.linspace(0,21,22)
# A3s = np.linspace(0,4,5)

alpha = 1
beta = 2
# omega0s = [10]
omega0s = np.linspace(4,20,16*10+1)
phi2 = 0 
phi3s = np.linspace(0,2, 41)
phi3s = [round(i,2) for i in phi3s]


centres = [1,2]
funcs = [Cosine, Cosine]

# dfnameOld = "TriangleRatios-v2.csv"
# dfO = pd.read_csv(dataLoc+dfnameOld, 
#                  index_col=False)

onsite2 = 0
onsite3 = 0



 
i = len(dfN)

for A2 in reversed(A2s):
        
    for A3 in reversed(A3s):
        
        A3start = time.time()
        # print( alpha, beta, A2, A3)
        for omega0 in omega0s:
            omega0 = np.round(omega0, 2)
            
            
            print(A2, A3, omega0)
            for phi3_frac in phi3s:
                phi3_frac = np.round(phi3_frac, 3)
                
                phi3 = pi*phi3_frac
        
                omega2 = alpha*omega0
                omega3 = beta*omega0
            
                T = 2*pi/omega0
            
            
                # first term expansion term
                J23_real = -(1/T)*integrate.quad(lambda t: cos(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
                J23_imag = -1j*(1/T)*integrate.quad(lambda t: sin(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
                # we are removing esimate of absolute error
                J23 = J23_real + J23_imag
                
                J31_real = -(1/T)*integrate.quad(lambda t: cos(-A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
                J31_imag = -1j*(1/T)*integrate.quad(lambda t: sin(-A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
                J31 = J31_real + J31_imag
            
                J12 = -jv(0,A2/omega2)
                
                # HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])
                HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])

                
                #full Hamiltonian evolution
                paramss = [[A2, omega2, 0, 0], [A3, omega3, phi3, 0]]
                _, HF = CreateHFGeneralLoopA(3, centres, funcs, paramss, T, 1)
                
                if not np.isnan(HF).all():
                    for site in range(3):
                        HF = RemoveWannierGauge(HF, site, 3)
                        HF_FT = RemoveWannierGauge(HF_FT, site, 3)
                    
                    
                        
                    J12 = HF_FT[1,0] # should be real?
                    J23 = HF_FT[2,1] # should be real ?
                    J31 = HF_FT[0,2]
    
                    J12_Ham = HF[1][0] # should be real?
                    J23_Ham = HF[2][1]
                    J31_Ham = HF[0][2]
                    
                    
                    O1 = HF[0][0]
                    O2 = HF[1][1]
                    O3 = HF[2][2]
        
                    dfN.loc[i] = [np.float32(A2), np.float32(A3), np.float32(omega0), np.uint32(alpha), np.uint32(beta)
                                  , np.float32(phi3_frac), 
                                  np.complex128(J12), np.complex128(J23), np.complex128(J31), 
                                  np.complex128(J12_Ham), np.complex128(J23_Ham), np.complex128(J31_Ham),
                                  np.complex128(O1), np.complex128(O2), np.complex128(O3)]
                    
                    
                    
                else:
                    dfN.loc[i] = [A2, A3, omega0, alpha, beta, phi3_frac, 
                                  np.nan, np.nan, np.nan, 
                                  np.nan, np.nan, np.nan,
                                  np.nan, np.nan, np.nan]
                i +=1
        
        # dfN['A2'] = dfN['A2'].apply(np.real)
        # dfN['A3'] = dfN['A3'].apply(np.real)
        # dfN['omega0'] = dfN['omega0'].apply(np.real)
        # dfN['alpha'] = dfN['alpha'].apply(np.real)
        # dfN['beta'] = dfN['beta'].apply(np.real)
        # dfN['phi3/pi'] = dfN['phi3/pi'].apply(np.real)
        
        # dfO = dfO.append(dfN, ignore_index=True, sort=False)
        dfN = dfN.astype({
            'A2': np.float32,
                          'A3': np.float32,
                          'omega0': np.float32,
                          "alpha":np.uint8,
                           "beta":np.uint8,
                           "phi3/pi":np.float32,
                         "FT-J12":np.complex128,
                         "FT-J23":np.complex128,
                         "FT-J31":np.complex128,
                         "HE-J12":np.complex128,
                         "HE-J23":np.complex128,
                         "HE-J31":np.complex128,
                         "HE-O1":np.complex128,
                         "HE-O2":np.complex128,
                         "HE-O3":np.complex128
                         })
        
        dfN.to_csv(dataLoc+dfname,
                  index=False, 
                  # columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31"]
                  )
        
        A3end = time.time()
        print("   ", np.round(A3end - A3start, 1), "s")
        
        

# dfN['A2'] = dfN['A2'].apply(np.real)
# dfN['A3'] = dfN['A3'].apply(np.real)
# dfN['omega0'] = dfN['omega0'].apply(np.real)
# dfN['alpha'] = dfN['alpha'].apply(np.real)
# dfN['beta'] = dfN['beta'].apply(np.real)
# dfN['phi3/pi'] = dfN['phi3/pi'].apply(np.real)

dfN = dfN.astype({'A2': np.float32,
                         'A3': np.float32,
                         'omega0': np.float32,
                         "alpha":np.uint32,
                         "beta":np.uint32,
                         "phi3/pi":np.float32,
                         "FT-J12":np.complex128,
                         "FT-J23":np.complex128,
                         "FT-J31":np.complex128,
                         "HE-J12":np.complex128,
                         "HE-J23":np.complex128,
                         "HE-J31":np.complex128,
                         "HE-O1":np.complex128,
                         "HE-O2":np.complex128,
                         "HE-O3":np.complex128
                         })


dfN.to_csv(dataLoc+dfname,
                  index=False, 
                  # columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31"]
                  )



