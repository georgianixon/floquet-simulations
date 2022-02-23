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
from hamiltonians import Cosine

dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/OldStuff/"
dfname = "TriangleRatios.csv"

sns.set(style="darkgrid")
sns.set(rc={'axes.facecolor':'0.96'})
size=12
params = {
            'legend.fontsize': size*0.7,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.7,
          'ytick.labelsize': size*0.7,
          'font.size': size,
          'font.family': 'STIXGeneral',
#          'axes.titlepad': 25,
          'mathtext.fontset': 'stix',
          
          # 'axes.facecolor': 'white',
          'axes.edgecolor': 'white',
          'axes.grid': True,
          'grid.alpha': 1,
          # 'grid.color': "0.9"
          "text.usetex": True
          }


mpl.rcParams.update(params)
mpl.rcParams["text.latex.preamble"] = mpl.rcParams["text.latex.preamble"] + r'\usepackage{xfrac}'

CB91_Blue = 'darkblue'#'#2CBDFE'
oxfordblue = "#061A40"
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
red = "#FC4445"
newred = "#E4265C"
flame = "#DD6031"

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
               CB91_Purple,
                # CB91_Violet,
                'dodgerblue',
                'slategrey', newred]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)



#%%
A2s = np.linspace(0, 30, 31)
A3s = np.linspace(0, 30, 31)

alpha = 2
beta = 9
omega0s = np.linspace(2,20,18*10+1)



dfO = pd.read_csv(dataLoc+dfname, 
                 index_col=False)



dfN = pd.DataFrame(columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31", "n1", "n2", "n3"])
i = 0
for j, (alpha, beta, A2, A3) in enumerate(notDone):
    print(j, alpha, beta, A2, A3)
    for omega0 in omega0s:
        # print(omega0)
    
        
        omega2 = alpha*omega0
        omega3 = beta*omega0
    
        T = 2*pi/omega0
    
    
    
        J23_real = integrate.quad(lambda t: cos(A3/omega3*sin(omega3*t) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
        
        J23_imag = 1j*integrate.quad(lambda t: sin(A3/omega3*sin(omega3*t) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
        # we are removing esimate of absolute error
        J23 = np.abs(omega0/2/pi*(J23_real + J23_imag))
        J12 = jv(0,A2/omega2)
        J31 = jv(0,A3/omega3)
        
        n1 = J12 / J31
        n2 = J23 / J12
        n3 = J31 / J23
        
        dfN.loc[i] = [A2, A3, omega0, alpha, beta, J12, J23, J31, n1, n2, n3]
        i +=1

dfO = dfO.append(dfN, ignore_index=True, sort=False)
dfO.to_csv(dataLoc+dfname,
                  index=False, 
                  columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31", "n1", "n2", "n3"])

