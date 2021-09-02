# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:45:45 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"

from numpy.linalg import eig
from numpy import  pi, log, exp, sin
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd 
import time
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/floquet-simulations/src')
from hamiltonians import  CreateHF, SolveSchrodinger


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
        

def ConvertComplex(s):
    return np.complex128(s.replace('i', 'j').replace('*I', 'j').replace('*^', 'e'))

sh = "/Users/"+place+"/Code/MBQD/floquet-simulations/"
dfname = "data/analysis-G.csv"


# df = pd.DataFrame(columns=["form", "rtol","N", 
#                            "a1", "a2", 
#                            "omega1", "omega multiplier",
#                            "phi1", "phi offset",
#                             "square",
#                             "chi",
#                             "gamma",
#                             "triangle",
#                             "alpha",
#                             "tilde",
#                             "star",
#                             "beta",
#                             "rho",
#                             "epsilon",
#                             "delta"])
    
# df.to_csv(sh+dfname,
#                    index=False, 
#                    columns=['form', 'rtol', 'N', 
#                             'a1', "a2", 
#                             "omega1", "omega multiplier",
#                            "phi1", "phi offset",
#                             "square",
#                             "chi",
#                             "gamma",
#                             "triangle",
#                             "alpha",
#                             "tilde",
#                             "star",
#                             "beta",
#                             "rho",
#                             "epsilon",
#                             "delta"]
#                    )

#%%
df_dtype_dict = {'form':str, "rtol":np.float64, 'N':int,
                 'a1':np.float64, 'a2':np.float64,
                 'omega1':np.float64, 'omega multiplier':np.float64,
                 'phi1':np.float64, 'phi offset':np.float64,
                 "square": np.complex128,
                    "chi": np.complex128,
                    "gamma": np.complex128,
                    "triangle": np.complex128,
                    "alpha": np.complex128,
                    "tilde": np.complex128,
                    "star": np.complex128,
                    "beta": np.complex128,
                    "rho": np.complex128,
                    "epsilon": np.complex128,
                    "delta": np.complex128}

df = pd.read_csv(sh+dfname, 
                 index_col=False, 
                 converters={"square": ConvertComplex,
                            "chi": ConvertComplex,
                            "gamma": ConvertComplex,
                            "triangle": ConvertComplex,
                            "alpha": ConvertComplex,
                            "tilde": ConvertComplex,
                            "star": ConvertComplex,
                            "beta": ConvertComplex,
                            "rho": ConvertComplex,
                            "epsilon": ConvertComplex,
                            "delta": ConvertComplex,
                            })



#%%

 # need tp dp 1e-6 phi = 0
N = 51; 
centre=25;
form='SS-p' 

rtol = 1e-11
a = 500
# phis = [ pi/7, pi/6, pi/5, pi/4, pi/3, pi/2, 0]
phis = [ pi/5, pi/4,  pi/3, pi/2, 0]
phiOffset = 0
omegaMultiplier = 0
onsite = 0



#def RGaugeMatrix(N, centre, a, omega, phi):
#    matrix = np.zeros((N, N), dtype=np.complex128)
#    np.fill_diagonal(matrix, 1)  
#    matrix[centre][centre] = exp(-1j*a*sin(phi)/omega)
#    return matrix



for phi in phis:
    print('a=',a,'  phi=',phi)
    df1 = pd.DataFrame(columns=["form", "rtol", "N", 
                                "a1", 
                                "a2",
                                "omega1",
                                "omega multiplier",
                                "phi1",
                                "phi offset",
                                "square",
                                "chi",
                                "gamma",
                                "triangle",
                                "alpha",
                                "tilde",
                                "star",
                                "beta",
                                "rho",
                                "epsilon",
                                "delta"])
    for i, omega1 in enumerate(np.linspace(3.1, 20, int((20-3.1)*10+1), endpoint=True)):
        
        start = time.time()
        
        omega1 = round(omega1, 1)
        print(omega1)
        
        if form == "SS-p":
            aInput = a
            omegaInput = omega1
            phiInput = phi
            onsiteInput = onsite
            
        # elif form =="DS-p" or form == "SSDF-p":
        #     omega2 = omegaMultiplier*omega1
        #     aInput = [a1,a2]
        #     omegaInput = [omega1,omega2]
        #     phiInput = [phi, phi+phiOffset]

        # calculate effective Hamiltonian 
        UT, HF = CreateHF(form, rtol, N, centre, aInput, omegaInput, phiInput, onsiteInput)
        

        
        # log matrix elements
        square = HF[centre-2][centre]
        chi = HF[centre-1][centre-1]
        gamma = HF[centre-1][centre]
        triangle = HF[centre-1][centre+1]
        alpha = HF[centre][centre]
        tilde = HF[centre][centre+1]
        star = HF[centre][centre+2]
        beta = HF[centre+1][centre+1]
        rho = HF[centre+1][centre+2]
        epsilon = HF[centre+1][centre+3]
        delta = HF[centre+2][centre+2]
        
        
        print('   ',time.time()-start, 's')
        
        df1.loc[i] = [form, 
                      rtol,
                      N,
                      a,
                      0,
                      omega1,
                      omegaMultiplier,
                      phi,
                      phiOffset,
                      square,
                      chi,
                      gamma,
                      triangle,
                      alpha,
                      tilde,
                      star,
                      beta,
                      rho,
                      epsilon,
                      delta]


    df = df.append(df1, ignore_index=True, sort=False)
    df= df.astype(dtype=df_dtype_dict)
    
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
    df.to_csv(sh+dfname,
              index=False, 
              columns=['form', 'rtol','N',
                       'a1', 'a2', 
                       'omega1', 'omega multiplier', 
                       'phi1', "phi offset",
                      "square",
                        "chi",
                        "gamma",
                        "triangle",
                        "alpha",
                        "tilde",
                        "star",
                        "beta",
                        "rho",
                        "epsilon",
                        "delta"])
