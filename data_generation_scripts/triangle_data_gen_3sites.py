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
beta = 2
gamma = 3
omega0 = 8

omega1 = alpha*omega0
omega2 = beta*omega0
omega3 = gamma*omega0

# a = np.linspace(0,2,21)


phi3_fracs = np.array([round(i, 2) for i in np.linspace(0,2,51)])

# phi3_fracs = np.array([round(i,2) for i in b if i not in a]) 



df_dir = Path(__file__).absolute().parent.parent/"paper_data"/f"Heff_omega={omega0},alpha={alpha},beta={beta},gamma={gamma}.csv"
df_dir_save = df_dir
print(df_dir)
if not os.path.isfile(df_dir):
    dfN = pd.DataFrame(columns=["A1", "A2", "A3", "omega0", "alpha", "beta", "gamma", "phi2/pi", "phi3/pi",
                            "FT-J12", "FT-J23", "FT-J31", 
                            # "HE-J12", "HE-J23", "HE-J31",
                            # "HE-O1", "HE-O2", "HE-O3" 
                            "FT-LowerT.X",
				                    "FT-LowerT.Y",
                                    "xi"
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



dfN = dfN.astype({"A1":np.float64,
                    'A2': np.float64,
                          'A3': np.float64,
                          'omega0': np.float64,
                          "alpha":np.uint8,
                          "beta":np.uint8,
                          "gamma":np.uint8,
                          "phi2/pi":np.float64,
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

                        "FT-LowerT.X":np.float64,
				                "FT-LowerT.Y":np.float64,
                                "xi":np.float64
                          })


# A2s = np.append(np.linspace(0,40,401), [round(jn_zeros(0,1)[0]*omega0, 6)])
# A3s = np.append(np.linspace(0,40,401), [round(jn_zeros(0,1)[0]*omega0, 6)])
# A2s = np.append(np.linspace(0,19.2,193), [round(jn_zeros(0,1)[0]*omega0, 6)])
# A3s = np.append(np.linspace(0,19.2,193), [round(jn_zeros(0,1)[0]*omega0, 6)])
# A2s = np.append(np.linspace(0,18.5,1850+1), [round(jn_zeros(0,1)[0]*omega0, 6)])
# A3s = np.linspace(37,38.5,150+1)

lst1 = np.linspace(0,45,91)
# lst2 = np.linspace(0,45,46)
# lst2 = np.linspace(0.3, 44.3, 89)
# A2s = np.array([round(i,2) for i in lst1 if i not in lst2])
# A3s = lst1
lst3 = np.linspace(45,70,51)
lst2 = np.linspace(0,70,71)

A1s = np.array([round(i,2) for i in lst2])
A2s = np.array([round(i,2) for i in lst2])
A3s = np.array([round(i,2) for i in lst2])

# A2s = np.linspace(41.55, 41.7, 16)
# A3s = [21]

# A2s = A2s[A2s<39]
# A3s = np.array([round(i,2) for i in np.sort(np.concatenate((lst1, lst2)))])
# A3s = np.array([round(i,2) for i in lst1 if i not in lst2])
# A2s = A2s[A2s<8]
centres = [1,2,3]
funcs = [Cosine, Cosine,Cosine]
 
i = len(dfN)

for A1 in reversed(A1s):
  for A2 in reversed(A2s):
      A2start = time.time()
      for A3 in reversed(A3s):
          print(A1, A2, A3)
          for phi3_frac in phi3_fracs:
              for phi2_frac in phi3_fracs:
                phi3 = pi*phi3_frac
                phi2 = pi*phi2_frac
            
                T = 2*pi/omega0

                J12_real =    (1/T)*integrate.quad(lambda t: cos(A2/omega2*sin(omega2*t + phi2) - A1/omega1*sin(omega1*t)), -T/2, T/2)[0]
                J12_imag = 1j*(1/T)*integrate.quad(lambda t: sin(A2/omega2*sin(omega2*t + phi2) - A1/omega1*sin(omega1*t)), -T/2, T/2)[0]
                J12 = J12_real + J12_imag

                # first term expansion term
                J23_real =    (1/T)*integrate.quad(lambda t: cos(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t + phi2)), -T/2, T/2)[0]
                J23_imag = 1j*(1/T)*integrate.quad(lambda t: sin(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t + phi2)), -T/2, T/2)[0]
                # we are removing esimate of absolute error
                J23 = J23_real + J23_imag

                J31_real =    (1/T)*integrate.quad(lambda t: cos(A1/omega1*sin(omega1*t) - A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
                J31_imag = 1j*(1/T)*integrate.quad(lambda t: sin(A1/omega1*sin(omega1*t) - A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
                J31 = J31_real + J31_imag
                
                # HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])
                HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])

                xi = np.angle(J12*J23*J31)

                # full Hamiltonian evolution
                # paramss = [[A2, omega2, 0, 0], [A3, omega3, phi3, 0]]
                # _, HF = CreateHFGeneralLoopA(3, centres, funcs, paramss, T, 1)
                    
                    
                J12_FT = HF_FT[1,0] 
                J23_FT = HF_FT[2,1] 
                J31_FT = HF_FT[0,2]

                J12_FT_abs = np.abs(J12_FT)
                J23_FT_abs = np.abs(J23_FT)
                J31_FT_abs = np.abs(J31_FT)

                R1223_FT = J12_FT_abs/J23_FT_abs
                R3123_FT = J31_FT_abs/J23_FT_abs
                R3112_FT = J31_FT_abs/J12_FT_abs
                R2312_FT = J23_FT_abs/J12_FT_abs
                R1231_FT = J12_FT_abs/J31_FT_abs
                R2331_FT = J23_FT_abs/J31_FT_abs   
                        
                lowerTriangle_X_FT, lowerTriangle_Y_FT = ListRatioLowerTriangle(R1223_FT, 
                        R3123_FT, R2312_FT, R3112_FT, R1231_FT, R2331_FT)
                    

                dfN.loc[i] = [f"{A1:.6f}", f"{A2:.6f}", f"{A3:.6f}", np.float64(omega0), np.uint32(alpha), np.uint32(beta), np.uint32(gamma)
                              , np.float64(phi2_frac), np.float64(phi3_frac),
                              np.complex128(J12), np.complex128(J23), np.complex128(J31), 
                            #   np.complex128(J12_Ham), np.complex128(J23_Ham), np.complex128(J31_Ham),
                            #   np.complex128(O1), np.complex128(O2), np.complex128(O3),
                            np.float64(lowerTriangle_X_FT), np.float64(lowerTriangle_Y_FT),
                            xi
                              ]
                    
                    
                i +=1
              
              # dfN['A2'] = dfN['A2'].apply(np.real)
              # dfN['A3'] = dfN['A3'].apply(np.real)
              # dfN['omega0'] = dfN['omega0'].apply(np.real)
              # dfN['alpha'] = dfN['alpha'].apply(np.real)
              # dfN['beta'] = dfN['beta'].apply(np.real)
              # dfN['phi3/pi'] = dfN['phi3/pi'].apply(np.real)
              
              # dfO = dfO.append(dfN, ignore_index=True, sort=False)
      # dfN = dfN.astype({
      #    "A1"
      #     'A2': np.float32,
      #                   'A3': np.float64,
      #                   'omega0': np.float64,
      #                   "alpha":np.uint32,
      #                     "beta":np.uint32,
      #                     "phi3/pi":np.float64,
      #                   "FT-J12":np.complex128,
      #                   "FT-J23":np.complex128,
      #                   "FT-J31":np.complex128,
      #                 #  "HE-J12":np.complex128,
      #                 #  "HE-J23":np.complex128,
      #                 #  "HE-J31":np.complex128,
      #                 #  "HE-O1":np.complex128,
      #                 #  "HE-O2":np.complex128,
      #                 #  "HE-O3":np.complex128
      #                 "FT-LowerT.X":np.float64,
      #                 "FT-LowerT.Y":np.float64,
      #                 "xi":np.float64
      #                   })
      
          dfN.to_csv(df_dir_save,
                    index=False, 
                    # columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31"]
                    )
      
      A2end = time.time()
      print("   ", np.round(A2end - A2start, 1), "s")

  # dfN = dfN.groupby(by=["A1", "A2", "A3", "omega0", "alpha", "beta", "gamma", "phi2/pi", "phi3/pi" ]).agg({'FT-J12':"mean", 
  #                                             'FT-J23':"mean", 
  #                                             'FT-J31':"mean", 
  #                                             "FT-LowerT.X":"mean",
  #                                             "FT-LowerT.Y":"mean",
  #                                                       "xi":"mean"
  #                           }).reset_index()

  # dfN = dfN.astype({
  #         'A2': np.float32,
  #                       'A3': np.float64,
  #                       'omega0': np.float64,
  #                       "alpha":np.uint32,
  #                         "beta":np.uint32,
  #                         "phi3/pi":np.float64,
  #                       "FT-J12":np.complex128,
  #                       "FT-J23":np.complex128,
  #                       "FT-J31":np.complex128,
  #                     #  "HE-J12":np.complex128,
  #                     #  "HE-J23":np.complex128,
  #                     #  "HE-J31":np.complex128,
  #                     #  "HE-O1":np.complex128,
  #                     #  "HE-O2":np.complex128,
  #                     #  "HE-O3":np.complex128
  #                     "FT-LowerT.X":np.float64,
  #                     "FT-LowerT.Y":np.float64,
  #                     "xi":np.float64
  #                       })
      
  # dfN.to_csv(df_dir,
  #               index=False, 
  #               # columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31"]
  #               )