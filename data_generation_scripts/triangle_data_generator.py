# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:56:45 2022

@author: Georgia
"""
from scipy.special import jv, jn_zeros
import numpy as np
import scipy.integrate as integrate
from numpy import pi, exp, sin, cos
import pandas as pd
import os
from pathlib import Path

from floquet_simulations.hamiltonians import CreateHFGeneral, ConvertComplex, ListRatioLowerTriangle
from floquet_simulations.periodic_functions import Cosine
import time

alpha = 1
beta = 1
omega0 = 8

df_dir = Path(__file__).absolute().parent.parent/"paper_data"/f"Heff_omega={omega0},alpha={alpha},beta={beta},phi3=0,2.csv"
print(df_dir)
if not os.path.isfile(df_dir):
    dfN = pd.DataFrame(columns=["A2", "A3", "omega0", "alpha", "beta",  "phi3/pi",
                            "FT-J12", "FT-J23", "FT-J31", 
                            # "HE-J12", "HE-J23", "HE-J31",
                            # "HE-O1", "HE-O2", "HE-O3" 
                            # "FT-LowerT.X",
				                    # "FT-LowerT.Y",
                              ])
else:
    dfN = pd.read_csv(df_dir,
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



dfN = dfN.astype({'A2': np.float64,
                          'A3': np.float64,
                          'omega0': np.float64,
                          "alpha":np.uint8,
                          "beta":np.uint8,
                          "phi3/pi":np.float64,
                          "FT-J12":np.complex128,
                          "FT-J23":np.complex128,
                          "FT-J31":np.complex128,
                        #   "HE-J12":np.complex128,
                        #   "HE-J23":np.complex128,
                        #   "HE-J31":np.complex128,
                        #   "HE-O1":np.complex128,
                        #   "HE-O2":np.complex128,
                        #   "HE-O3":np.complex128
                        # "FT-LowerT.X":np.float64,
				                # "FT-LowerT.Y":np.float64
                          })

# A2s = np.linspace(0,4,5)
A2s = np.append(np.linspace(0,19.2,193), [round(jn_zeros(0,1)[0]*omega0, 6)])
A3s = np.append(np.linspace(0,19.2,193), [round(jn_zeros(0,1)[0]*omega0, 6)])


phi3_frac=0


centres = [1,2]
funcs = [Cosine, Cosine]

# dfnameOld = "TriangleRatios-v2.csv"
# dfO = pd.read_csv(dataLoc+dfnameOld, 
#                  index_col=False)


 
i = len(dfN)

for A2 in reversed(A2s):
    print(A2)
    A3start = time.time()
    for A3 in reversed(A3s):
        
       
        # print( alpha, beta, A2, A3)

        # omega0 = np.round(omega0, 2)
        
        
  
            
        phi3 = pi*phi3_frac
    
        omega2 = alpha*omega0
        omega3 = beta*omega0
    
        T = 2*pi/omega0
    
    
        # first term expansion term
        # J23_real = -(1/T)*integrate.quad(lambda t: cos(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
        # J23_imag = -1j*(1/T)*integrate.quad(lambda t: sin(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
        # we are removing esimate of absolute error
        # J23 = J23_real + J23_imag

        J23 = -jv(0, (A3-A2)/omega0)

        J31 = -jv(0, A3/omega3)
    
        J12 = -jv(0,A2/omega2)
        
        # HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])
        HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])

        
        #full Hamiltonian evolution
        # paramss = [[A2, omega2, 0, 0], [A3, omega3, phi3, 0]]
        # _, HF = CreateHFGeneralLoopA(3, centres, funcs, paramss, T, 1)
        
        # if not np.isnan(HF).all():
            # for site in range(3):
                # HF = RemoveWannierGauge(HF, site, 3)
                # HF_FT = RemoveWannierGauge(HF_FT, site, 3)
            
            
                
        J12_FT = HF_FT[1,0] # should be real?
        J23_FT = HF_FT[2,1] # should be real ?
        J31_FT = HF_FT[0,2]


    
        # J12_FT_abs = np.abs(J12_FT)
        # J23_FT_abs = np.abs(J23_FT)
        # J31_FT_abs = np.abs(J31_FT)

        # R1223_FT = J12_FT_abs/J23_FT_abs
        # R3123_FT = J31_FT_abs/J23_FT_abs
        # R3112_FT = J31_FT_abs/J12_FT_abs
        # R2312_FT = J23_FT_abs/J12_FT_abs
        # R1231_FT = J12_FT_abs/J31_FT_abs
        # R2331_FT = J23_FT_abs/J31_FT_abs
                
            
                
        # lowerTriangle_X_FT, lowerTriangle_Y_FT = ListRatioLowerTriangle(R1223_FT, 
        #         R3123_FT, R2312_FT, R3112_FT, R1231_FT, R2331_FT)
            

        dfN.loc[i] = [f"{A2:.6f}", f"{A3:.6f}", np.float64(omega0), np.uint32(alpha), np.uint32(beta)
                      , np.float64(phi3_frac), 
                      np.complex128(J12), np.complex128(J23), np.complex128(J31), 
                    #   np.complex128(J12_Ham), np.complex128(J23_Ham), np.complex128(J31_Ham),
                    #   np.complex128(O1), np.complex128(O2), np.complex128(O3),
                    # np.float64(lowerTriangle_X_FT), np.float64(lowerTriangle_Y_FT)
                      ]
            
            
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
                    # "FT-LowerT.X":np.float64,
                    # "FT-LowerT.Y":np.float64
                      })
    
    dfN.to_csv(df_dir,
              index=False, 
              # columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31"]
              )
    
    A3end = time.time()
    print("   ", np.round(A3end - A3start, 1), "s")
