# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:08:09 2020

@author: Georgia
"""

"""
Create csv that gives hopping as a function of a, omega, type of hamiltonian,
and other parameters
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
from hamiltonians import  create_HF, solve_schrodinger


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
    return np.complex(s.replace('i', 'j').replace('*I', 'j').replace('*^', 'e'))

sh = "/Users/"+place+"/Code/MBQD/floquet-simulations/"
dfname = "data/analysis-G-withnewgauge.csv"


# df = pd.DataFrame(columns=["form", "rtol",
#                                     "a", 
#                                     "omega", "phi", "N", 
#                                     "hopping",
#                                     "hopping back",
#                                     "onsite",
#                                     "next onsite",
#                                     "NNN overtop",
#                                     "NNN star",
#                                     "NNN square"])
    
# df.to_csv(sh+dfname,
#                   index=False, 
#                   columns=['form', 'rtol', 'a', 'omega', 'phi',
#                           'N', "hopping",
#                                     "hopping back",
#                                     "onsite",
#                                     "next onsite",
#                                     "NNN overtop",
#                                     "NNN star",
#                                     "NNN square"])

#%%
df_dtype_dict = {'form':str, "rtol":np.float64,
                 'a':np.float64, 
            'omega':np.float64, 'phi':np.float64, 'N':int,
            'hopping':np.complex128,
            'hopping back':np.complex128,
            'onsite':np.complex128, 
            'next onsite':np.complex128,'NNN overtop':np.complex128,
            'NNN star':np.complex128, 'NNN square':np.complex128 }

df = pd.read_csv(sh+dfname, 
                 index_col=False, 
                 converters={'hopping': convert_complex,
                             'hopping back': convert_complex,
                             'onsite':convert_complex,
                             'next onsite':convert_complex, 
                             'NNN overtop':convert_complex,
                             'NNN star':convert_complex,
                             'NNN square':convert_complex,
                                              })



#%%

 # need tp dp 1e-6 phi = 0
N = 51; 
centre=25;
form='SS-p' 
rtol = 1e-11
aas = [35]
phis = [ pi/7, pi/6, pi/5, pi/4, pi/3, pi/2, 0]
# phis = [ pi/2, 0]



def RGaugeMatrix(N, centre, a, omega, phi):
    matrix = np.zeros((N, N), dtype=np.complex128)
    np.fill_diagonal(matrix, 1)  
    matrix[centre][centre] = exp(-1j*a*sin(phi)/omega)
    return matrix



for a in aas:
    for phi in phis:
        print('a=',a,'  phi=',phi)
        df1 = pd.DataFrame(columns=["form", "rtol",
                                    "a", 
                                    "omega", "phi", "N", 
                                    "hopping",
                                    "hopping back",
                                    "onsite",
                                    "next onsite",
                                    "NNN overtop",
                                    "NNN star",
                                    "NNN square"])
        for i, omega in enumerate(np.linspace(20.1, 200, 10*180, endpoint=True)):
            omega = round(omega, 1)
            print(omega)
            
            start = time.time()
            """
            HF
            """  
            UT, HF = create_HF(form, rtol, N, centre, a,phi, omega)
            
            R = RGaugeMatrix(N, centre, a, omega, phi)
            HF = np.dot(np.conj(R.T), np.dot(HF, R))
            
            
            hopping=HF[centre][centre+1]
            hoppingBack=HF[centre-1][centre]
            onsite = HF[centre][centre]
            next_onsite=HF[centre+1][centre+1]
            NNN_overtop=HF[centre-1][centre+1]
            NNNStar = HF[centre][centre+2]
            NNNSquare = HF[centre-2][centre]
            
            
            print('   ',time.time()-start, 's')
            
            df1.loc[i] = [form, 
                          rtol,
                          a,
                          omega,
                          phi, 
                          N,
                          hopping,
                          hoppingBack,
                          onsite,
                          next_onsite,
                          NNN_overtop,
                          NNNStar,
                          NNNSquare]

    
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
                  columns=['form', 'rtol', 'a', 'omega', 'phi',
                          'N', 'hopping', "hopping back",
                          "onsite", "next onsite", 
                                    "NNN overtop",
                                    "NNN star",
                                    "NNN square"])
    
