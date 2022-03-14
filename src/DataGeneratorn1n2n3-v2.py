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
from hamiltonians import CreateHFGeneral
from hamiltonians import Cosine,  RemoveWannierGauge

dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/OldStuff/"
dfname = "TriangleRatios-v2-1.csv"


#%%

A2s = np.linspace(0,30,11)
A3s = np.linspace(0,30,11)


alpha = 1
beta = 2
# omega0s = [10]
omega0s = np.linspace(4,20,16*2+1)
phi2 = 0 
phi3s = np.linspace(0,2, 21)
phi3s = [round(i,2) for i in phi3s]


centres = [1,2]
funcs = [Cosine, Cosine]

# dfnameOld = "TriangleRatios-v2.csv"
# dfO = pd.read_csv(dataLoc+dfnameOld, 
#                  index_col=False)

onsite2 = 0
onsite3 = 0


dfN = pd.DataFrame(columns=["A2", "A3", "omega0", "alpha", "beta",  "phi3/pi",
                            "FT-J12", "FT-J23", "FT-J31", 
                            "HE-J12", "HE-J23", "HE-J31",
                            "HE-O1", "HE-O2", "HE-O3"  ])

i = 0

for A2 in A2s:
    print(A2)
        
    for A3 in A3s:
        print("   ", A3)
        # print( alpha, beta, A2, A3)
        for omega0 in omega0s:
            # print(omega0)
            for phi3_frac in phi3s:
                phi3 = pi*phi3_frac
        
                omega2 = alpha*omega0
                omega3 = beta*omega0
            
                T = 2*pi/omega0
            
            
                # first term expansion term
                J23_real = (1/T)*integrate.quad(lambda t: cos(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
                J23_imag = 1j*(1/T)*integrate.quad(lambda t: sin(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
                # we are removing esimate of absolute error
                J23 = J23_real + J23_imag
                
                J31_real = (1/T)*integrate.quad(lambda t: cos(-A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
                J31_imag = 1j*(1/T)*integrate.quad(lambda t: sin(-A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
                J31 = J31_real + J31_imag
            
                J12 = jv(0,A2/omega2)
                
                # HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])
                HF_FT = np.array([[0,J12, np.conj(J31)], [np.conj(J12), 0, J23], [J31, np.conj(J23), 0]])

                
                #full Hamiltonian evolution
                paramss = [[A2, omega2, 0, 0], [A3, omega3, phi3, 0]]
                _, HF = CreateHFGeneral(3, centres, funcs, paramss, T, 1)
                
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
    
                dfN.loc[i] = [A2, A3, omega0, alpha, beta, phi3_frac, 
                              J12, J23, J31, 
                              J12_Ham, J23_Ham, J31_Ham,
                              O1, O2, O3]
                i +=1

dfN['A2'] = dfN['A2'].apply(np.real)
dfN['A3'] = dfN['A3'].apply(np.real)
dfN['omega0'] = dfN['omega0'].apply(np.real)
dfN['alpha'] = dfN['alpha'].apply(np.real)
dfN['beta'] = dfN['beta'].apply(np.real)
dfN['phi3/pi'] = dfN['phi3/pi'].apply(np.real)

# dfO = dfO.append(dfN, ignore_index=True, sort=False)
dfN.to_csv(dataLoc+dfname,
                  index=False, 
                  # columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31"]
                  )






