# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:46:24 2021

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
from hamiltonians import  CreateHF, SolveSchrodinger, ConvertComplex


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
        

sh = "/Users/"+place+"/Code/MBQD/floquet-simulations/"
dfname = "data/analysis-G-N6.csv"


df = pd.DataFrame(columns=["form", "rtol","N", 
                            "a", 
                            "omega", 
                            "phi", 
                            "onsite",
                            "O-3",
                            "O-2",
                            "O-1",
                            "O",
                            "O+1",
                            "O+2",
                            "O+3",
                            "N1-3",
                            "N1-2",
                            "N1-1",
                            "N1+1",
                            "N1+2",
                            "N1+3",
                            "N2-2",
                            "N2-1",
                            "N2",
                            "N2+1",
                            "N2+2",
                            "N3-2",
                            "N3-1",
                            "N3+1",
                            "N3+2",
                            "N4-1",
                            "N4",
                            "N4+1",
                            "N5-1",
                            "N5+1",
                            "N6"])
    
df.to_csv(sh+dfname,
                    index=False, 
                    columns=['form', 'rtol', 'N', 
                            'a',
                            "omega", 
                            "phi",
                            "onsite",
                            "O-3",
                            "O-2",
                            "O-1",
                            "O",
                            "O+1",
                            "O+2",
                            "O+3",
                            "N1-3",
                            "N1-2",
                            "N1-1",
                            "N1+1",
                            "N1+2",
                            "N1+3",
                            "N2-2",
                            "N2-1",
                            "N2",
                            "N2+1",
                            "N2+2",
                            "N3-2",
                            "N3-1",
                            "N3+1",
                            "N3+2",
                            "N4-1",
                            "N4",
                            "N4+1",
                            "N5-1",
                            "N5+1",
                            "N6"]
                    )

#%%
df_dtype_dict = {'form':str, "rtol":np.float64, 'N':int,
                 'a':np.float64, 
                 'omega':np.float64, 
                 'phi':np.float64,
                 "onsite":np.float64,
                 "O-3":np.float64,
                "O-2":np.float64,
                "O-1":np.float64,
                "O":np.float64,
                "O+1":np.float64,
                "O+2":np.float64,
                "O+3":np.float64,
                "N1-3":np.float64,
                "N1-2":np.float64,
                "N1-1":np.float64,
                "N1+1":np.float64,
                "N1+2":np.float64,
                "N1+3":np.float64,
                "N2-2":np.float64,
                "N2-1":np.float64,
                "N2":np.float64,
                "N2+1":np.float64,
                "N2+2":np.float64,
                "N3-2":np.float64,
                "N3-1":np.float64,
                "N3+1":np.float64,
                "N3+2":np.float64,
                "N4-1":np.float64,
                "N4":np.float64,
                "N4+1":np.float64,
                "N5-1":np.float64,
                "N5+1":np.float64,
                "N6":np.float64}

df = pd.read_csv(sh+dfname, 
                 index_col=False, 
                 converters={"O-3": ConvertComplex,
                            "O-2": ConvertComplex,
                            "O-1": ConvertComplex,
                            "O": ConvertComplex,
                            "O+1": ConvertComplex,
                            "O+2": ConvertComplex,
                            "O+3": ConvertComplex,
                            "N1-3": ConvertComplex,
                            "N1-2": ConvertComplex,
                            "N1-1": ConvertComplex,
                            "N1+1": ConvertComplex,
                            "N1+2": ConvertComplex,
                            "N1+3": ConvertComplex,
                            "N2-2": ConvertComplex,
                            "N2-1": ConvertComplex,
                            "N2": ConvertComplex,
                            "N2+1": ConvertComplex,
                            "N2+2": ConvertComplex,
                            "N3-2": ConvertComplex,
                            "N3-1": ConvertComplex,
                            "N3+1": ConvertComplex,
                            "N3+2": ConvertComplex,
                            "N4-1": ConvertComplex,
                            "N4": ConvertComplex,
                            "N4+1": ConvertComplex,
                            "N5-1": ConvertComplex,
                            "N5+1": ConvertComplex,
                            "N6": ConvertComplex
                            })



#%%

# need tp dp 1e-6 phi = 0
N = 51; 
centre=25;
form='SS-p' 

rtol = 1e-11
a = 15
phis = [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
onsite = 0
omegas = np.linspace(3.1, 20, int((20-3.1)*10+1), endpoint=True)



for phi in phis:
    print('a=',a,'  phi=',phi)
    df1 = pd.DataFrame(columns=["form", "rtol", "N", 
                                "a",
                                "omega",
                                "phi",
                                "onsite",
                                "O-3",
                                "O-2",
                                "O-1",
                                "O",
                                "O+1",
                                "O+2",
                                "O+3",
                                "N1-3",
                                "N1-2",
                                "N1-1",
                                "N1+1",
                                "N1+2",
                                "N1+3",
                                "N2-2",
                                "N2-1",
                                "N2",
                                "N2+1",
                                "N2+2",
                                "N3-2",
                                "N3-1",
                                "N3+1",
                                "N3+2",
                                "N4-1",
                                "N4",
                                "N4+1",
                                "N5-1",
                                "N5+1",
                                "N6"])
    for i, omega1 in enumerate(omegas):
        
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
        Om3 = HF[centre-3][centre-3]
        Om2 = HF[centre-2][centre-2]
        Om1 = HF[centre-1][centre-1]
        O = HF[centre][centre]
        Op1 = HF[centre+1][centre+1]
        Op2 = HF[centre+2][centre+2]
        Op3 = HF[centre+3][centre+3]
        
        N1m3 = HF[centre-3][centre-2]
        N1m2 = HF[centre-2][centre-1]
        N1m1 = HF[centre-1][centre]
        N1p1 = HF[centre][centre+1]
        N1p2 = HF[centre+1][centre+2]
        N1p3 = HF[centre+2][centre+3]
        
        N2m2 = HF[centre-3][centre-1]
        N2m1 = HF[centre-2][centre]
        N2 = HF[centre-1][centre+1]
        N2p1 = HF[centre][centre+2]
        N2p2 = HF[centre+1][centre+3]
        
        N3m2 = HF[centre-3][centre]
        N3m1 = HF[centre-2][centre+1]
        N3p1 = HF[centre-1][centre+2]
        N3p2 = HF[centre][centre+3]
        
        N4m1 = HF[centre-3][centre+1]
        N4 = HF[centre-2][centre+2]
        N4p1 = HF[centre-2][centre+3]
        
        N5m1 = HF[centre-3][centre+2]
        N5p1 = HF[centre-2][centre+1]
        
        N6 = HF[centre-3][centre+3]
        
        
        
        print('   ',time.time()-start, 's')
        
        df1.loc[i] = [form, 
                      rtol,
                      N,
                      a,
                      omega1,
                      phi,
                      onsite,
                      Om3,
                        Om2,
                        Om1,
                        O,
                        Op1,
                        Op2,
                        Op3,
                        
                        N1m3, 
                        N1m2,
                        N1m1, 
                        N1p1,
                        N1p2,
                        N1p3, 
                        
                        N2m2,
                        N2m1, 
                        N2 ,
                        N2p1, 
                        N2p2,
                        
                        N3m2, 
                        N3m1, 
                        N3p1,
                        N3p2, 
                        
                        N4m1, 
                        N4,
                        N4p1, 
                        
                        N5m1,
                        N5p1, 
                        
                        N6 
        ]


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
                       'a', 
                       'omega', 
                       'phi',
                       "onsite",
                      "O-3",
                            "O-2",
                            "O-1",
                            "O",
                            "O+1",
                            "O+2",
                            "O+3",
                            "N1-3",
                            "N1-2",
                            "N1-1",
                            "N1+1",
                            "N1+2",
                            "N1+3",
                            "N2-2",
                            "N2-1",
                            "N2",
                            "N2+1",
                            "N2+2",
                            "N3-2",
                            "N3-1",
                            "N3+1",
                            "N3+2",
                            "N4-1",
                            "N4",
                            "N4+1",
                            "N5-1",
                            "N5+1",
                            "N6"])