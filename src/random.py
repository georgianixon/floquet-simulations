# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:38:48 2022

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

dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/OldStuff/"
dfname = "TriangleRatios.csv"
import pandas as pd
dfO = pd.read_csv(dataLoc+dfname, 
                 index_col=False)

notDone = []
onesixone = []
for alpha in [1]:
    for beta in [2,3,4,5]:#[3,5,7,9]:
        # print(alpha, beta)
        for A2 in np.linspace(0,30, 31):
            for A3 in np.linspace(0,30,31):
                # print(alpha, beta, A, B)
                dfP = dfO[(dfO.beta == beta)
                          &(dfO.alpha == alpha)
                          &(dfO["A2"]==A2)
                          &(dfO["A3"]==A3)]
                if (len(dfP.omega0.unique()) != 161) & (len(dfP.omega0.unique()) != 181) :
                    if len(dfP.omega0.unique())==0:
                        # print(alpha, beta, A, B, len(dfP.omega0.unique()), dfP.omega0.min(), dfP.omega0.max())
                        notDone.append((alpha, beta, A2, A3))
                    else:
                        print(alpha, beta, A2, A3, len(dfP.omega0.unique()), dfP.omega0.min(), dfP.omega0.max())
                # else:
                #     onesixone.append((alpha, beta, A2, A3))


#%%

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

colourcycle = cycle(["darkblue", "#47DBCD", "#F3A0F2", "#E4265C", "#DD6031"])
colour = next(colourcycle)
delta = np.append(np.linspace(-1, -0.0001, 1000), np.linspace(0.0001,1,1000))
fig, ax = plt.subplots(figsize=(6,6))
for omega in reversed([0.00001, 0.001, 0.01, 0.1, 1]):


    # E1 = -delta + np.sqrt(np.square(delta) + np.abs(omega)**2)
    # E2 = -delta - np.sqrt(np.square(delta) + np.abs(omega)**2)
    # ax.plot(delta, E1, label="E+", color=colour)
    # ax.plot(delta, E2, label="E-", color=colour)
    
    c = omega**2/(2*omega**2 + 2*np.square(delta) - 2*delta*np.sqrt(np.square(delta) + np.abs(omega)**2))
    ax.plot(delta, c, '.', label=str(omega), color=colour)
    ax.legend()
    
    colour = next(colourcycle)
plt.show()

#%%


def PlotG(G):
    '''abs real imag'''
    
    apply = [
             np.abs, 
             np.real, np.imag]
    labels = [
              r'$\mathrm{Abs}\{G_{n,m}\}$', 
              r'$\mathrm{Re}\{G_{n,m}\}$',
              r'$\mathrm{Imag}\{G_{n,m}\}$'
              ]
    
    sz = 8
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                           figsize=(sz,sz/2))
    
    for n1, f in enumerate(apply):
        pcm = ax[n1].matshow(f(G), interpolation='none', cmap='PuOr',  norm=norm)
        ax[n1].set_title(labels[n1])
        ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[n1].set_xlabel('m')
    
    ax[0].set_ylabel('n', rotation=0, labelpad=10)
    cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    # fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
    fig.colorbar(pcm, cax=cax)
        
    #fig.savefig('', 
    #        format='pdf', bbox_inches='tight')
    plt.show()
    
#%%

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
from hamiltonians import CreateHFGeneral
from hamiltonians import Cosine, RemoveWannierGauge
dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/OldStuff/"
dfname = "TriangleRatios.csv"

dfO = pd.read_csv(dataLoc+dfname, 
                  index_col=False)



alpha = 2
beta = 3

A2 = 10
A3 = 30
omega0 = 5
omega2 = alpha*omega0
omega3 = beta*omega0
T = 2*pi/omega0;
onsite2 = 0; onsite3 = 0;
phi1 = 0; phi2 = 0

_, G = CreateHFGeneral(3, [1,2], [Cosine, Cosine], [[A2, alpha*omega0, phi1, onsite2], [A3, beta*omega0, phi2, onsite3]], T, circleBoundary = 1)
# for site in range(3):
#     G = RemoveWannierGauge(G, site, 3)

G[0,0]=0; G[1,1]=0;G[2,2]=0
J23_real = integrate.quad(lambda t: cos(A3/omega3*sin(omega3*t) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
J23_imag = 1j*integrate.quad(lambda t: sin(A3/omega3*sin(omega3*t) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
J23 = -omega0/2/pi*(J23_real + J23_imag)
J12 = -jv(0,A2/omega2)
J31 = -jv(0,A3/omega3)

G1 = np.array([[0,J12,0],[0,0,J23],[J31,0,0]])
G1 = G1 + np.conj(G1).T
# for site in range(3):
#     G1 = RemoveWannierGauge(G1, site, 3)

# we are removing esimate of absolute error
# J23 = np.abs(J23)


dfP = dfO[(dfO.beta == beta)
                    &(dfO.alpha == alpha)
                    &(dfO.omega0 == omega0)
                    &(dfO.A2 == A2)
                  &(dfO.A3 == A3)]


G1df =  np.array([[0,-dfP["J12"].values[0],0],[0,0,-dfP["J23"].values[0]],[-dfP["J31"].values[0],0,0]])
G1df = G1df + np.conj(G1df).T

# print(dfP["J12"].values[0], dfP["J23"].values[0], dfP["J31"].values[0])
# print(J12, J23, J31)
print(G1[0,1], G1[1,2], G1[0,2])
print(G[0,1],  G[1,2], G[0,2])
# print(np.abs(G[0,1]), J12, dfP["J12"].values[0],  np.abs(G[1,2]), J23, dfP["J23"].values[0], np.abs(G[0,2]), J31, dfP["J31"].values[0])



norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# linthresh = 1e-1
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
# 


PlotG(G)
PlotG(G1)
PlotG(G1df)


#%%

omegaMin = 10.1
omegaMax = 10.1
A2Min = 7
A2Max = 7
A3Min = 20
A3Max = 20
alpha = 2
beta = 5

df1 = dfO[(dfO.beta == beta)
                  &(dfO.alpha == alpha)
                  &(dfO.omega0 <= omegaMax)
                  &(dfO.omega0 >= omegaMin)
                  &(dfO.A2 >= A2Min)
                  &(dfO.A2 <= A2Max)
                  &(dfO.A3 >= A3Min)
                  &(dfO.A3 <= A3Max)]


df2 = dfO[(dfO.beta == beta)
                  &(dfO.alpha == alpha)
                  &(dfO.omega0 <= omegaMax)
                  &(dfO.omega0 >= omegaMin)
                  &(dfO.A2 >= A3Min)
                  &(dfO.A2 <= A3Max)
                  &(dfO.A3 >= A2Min)
                  &(dfO.A3 <= A2Max)]







