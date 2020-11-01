# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:35:42 2020

@author: Georgia
"""

from numpy.linalg import eig
import matplotlib.colors as col

from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import time
from hamiltonians import F_OSC

#%%

"""
For single site oscillation
"""

def create_HF_OSC(N, centre, a, phi, omega): 
    T=2*pi/omega
    tspan = (0,T)
    UT = np.zeros([N,N], dtype=np.complex_)
    start = time.time()
    for A_site_start in range(N):
    #    print(A_site_start)
        psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
        sol = solve_ivp(lambda t,psi: F_OSC(t, psi, N, 
                                             centre,
                                             a, 
                                             omega, 
                                             phi), 
                            tspan, psi0, rtol=1e-7, atol=1e-7)
        UT[:,A_site_start]=sol.y[:,-1]
    
    print(time.time()-start, 'seconds.')
    
    evals_U, evecs = eig(UT)
    evals_H = 1j / T *log(evals_U)
    HF = np.zeros([N,N], dtype=np.complex_)
    for i in range(N):
        term = evals_H[i]*np.outer(evecs[:,i], evecs[:,i])
        HF = HF+term
    return UT, HF

"""
Plot HF
"""

N=51; centre=25; a=35; phi=pi/7; omega=8.603205
UT, HF = create_HF_OSC(N, centre, a, phi, omega)
sz = 3
fig, ax = plt.subplots(figsize=(sz,sz))
norm = col.Normalize(vmin=-1, vmax=1)
# ax.matshow(np.abs(HF[centre-2:centre+3, centre-2:centre+3]), interpolation='none', cmap='PuOr', norm=norm)
ax.matshow(np.abs(HF[centre-2:centre+3, centre-2:centre+3]), interpolation='none', cmap='PuOr', norm=norm)

ax.tick_params(axis="x", bottom=True, top=False,  labelbottom=True, 
  labeltop=False)
ax.set_xlabel('m')
ax.set_ylabel('n', rotation=0, labelpad=10)

cax = plt.axes([1, 0.05, 0.06, 0.9])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
plt.show()

print(omega)
print(np.abs(HF[centre][centre+1]))