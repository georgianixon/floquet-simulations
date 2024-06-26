# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:08:09 2020

@author: Georgia
"""

"""
Create csv that gives hopping as a function of a, omega, type of hamiltonian,
and other parameters
"""

from numpy.linalg import eig
from numpy import  pi, log
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd 
import time
import sys
sys.path.append('/Users/Georgia/Code/MBQD/floquet-simulations/src')
from hamiltonians import  create_HF, solve_schrodinger

#%%

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
        

def convert_complex(s):
    return np.complex(s.replace('i', 'j'))

sh = '/Users/Georgia/Code/MBQD/floquet-simulations/'


df_dtype_dict = {'form':str, "rtol":np.float64,
                 'a':np.float64, 
            'omega':np.float64, 'phi':np.float64, 'N':int,
            'hopping':np.complex128,
            'onsite':np.complex128, 'next onsite':np.complex128,
            'NNN':np.complex128, 'NNN overtop':np.complex128}

df = pd.read_csv(sh+'data/analysis_gaus_complex.csv', 
                 index_col=False, 
                 converters={'hopping': convert_complex,
                             'onsite':convert_complex,
                             'next onsite':convert_complex,
                             'NNN':convert_complex, 
                             'NNN overtop':convert_complex,
                                              })


#%%

 # need tp dp 1e-6 phi = 0
N = 51; 
centre=25;
form='SS-p' 
rtol = 1e-7
aas = [30]
phis = [pi/4, pi/5, pi/6, pi/7, 0, pi/2, pi/3]
b = None
c = None

for a in aas:
    for phi in phis:
        print('a=',a,' b=',b,' c=',c,'  phi=',phi)
        df1 = pd.DataFrame(columns=["form", "rtol",
                                    "a", 
                                    "omega", "phi", "N", 
                                    "hopping", "onsite", 
                                    "next onsite", "NNN",
                                    "NNN overtop"])
        for i, omega in enumerate(np.linspace(3.7, 20, 164)):
            omega = round(omega, 1)
            print(omega)
            
            start = time.time()
            """
            HF
            """  
            UT, HF = create_HF(form, rtol, N, centre, a,b, c,phi, omega)
                
            """
            Localisation
            """
            psi0 = np.zeros(N, dtype=np.complex_); psi0[centre] = 1;
            tspan = (0, 10)
            sol = solve_schrodinger(form, rtol, N, centre,
                                    a, b, c, omega, phi,
                                    tspan, 100, psi0)
            localisation = np.sum(abs(sol[centre]))/101
            # localisation = np.nan
            
            hopping=HF[centre][centre+1]
            onsite = HF[centre][centre]
            next_onsite=HF[centre+1][centre+1]
            NNN = HF[centre][centre+2]
            NNN_overtop=HF[centre-1][centre+1]
            
            print('   ',time.time()-start, 's')
            
            df1.loc[i] = [form, 
                          rtol,
                          a,
                          omega,
                          phi, 
                          N,
                          hopping,
                          onsite,
                          next_onsite,
                          NNN,
                          NNN_overtop]

    
        df = df.append(df1, ignore_index=True, sort=False)
        df= df.astype(dtype=df_dtype_dict)
        
print('  grouping..')
df = df.groupby(by=['form', 'rtol', 'a', 'omega', 'phi', 
                 'N'], dropna=False).agg({
                        'hopping':filter_duplicates,
                        'onsite':filter_duplicates,
                        'next onsite':filter_duplicates,
                        'NNN':filter_duplicates,
                        'NNN overtop':filter_duplicates
                        }).reset_index()

print('   saving..')
df.to_csv(sh+'data/analysis_G.csv',
          index=False, 
          columns=['form', 'rtol', 'a', 'omega', 'phi',
                  'N', 'hopping', 
                  'onsite', 'next onsite', 'NNN',
                    'NNN overtop'])
    
