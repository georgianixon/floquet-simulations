# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:49 2020

@author: Georgia
"""


from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

from scipy.integrate import solve_ivp
from scipy.special import jv, jn_zeros
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
from hamiltonians import F_MG, F_OSC, create_HF, solve_schrodinger

import matplotlib
import seaborn as sns

size=15
params = {
            'legend.fontsize': 'small',
#          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size*1.2,
          'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
#          'axes.titlepad': 25
          'axes.grid':False,
          'font.family' : 'STIXGeneral',
          'mathtext.fontset':'stix'
          }
matplotlib.rcParams.update(params)

#%%


# choose particular HF

N = 51; A_site_start = 25;
centre = 25
a = 35;
b = np.nan
c = np.nan 
omega=10; 
phi=pi/7; T=2*pi/omega
tspan = (0,10)
Nt = 100
form = 'OSC'

t_eval = np.linspace(tspan[0], tspan[1], Nt)
psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;

sol = solve_schrodinger(form, N, centre, a, b, c, omega, phi, tspan, psi0, avg=0)

sz = 20
print(sol.y[25][0])
print(sol.y[25][-1])
fig, ax = plt.subplots(figsize=(sz,sz/2))
ax.matshow((np.abs(sol.y))**2, interpolation='none', cmap='Purples')
fig.colorbar(plt.cm.ScalarMappable(cmap='Purples'))
plt.show()

fig, ax = plt.subplots(figsize=(sz,sz/2))
ax.matshow(np.abs(sol.y), interpolation='none', cmap='Purples')
fig.colorbar(plt.cm.ScalarMappable(cmap='Purples'))
plt.show()

fig, ax = plt.subplots(figsize=(sz,sz/2))
ax.plot(sol.t, np.abs(sol.y)[25])
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                        figsize=(sz,sz/2))
ax[0].matshow(abs(sol.y)**2, interpolation='none', cmap='Purples')
ax[1].matshow(np.real(sol.y), interpolation='none', cmap='Purples')
ax[2].matshow(np.angle(sol.y), interpolation='none', cmap='Purples')
ax[0].set_title(r'$|\psi(t)|^2$')
ax[1].set_title(r'$\mathrm{Re}\{\psi(t)\}$')
ax[2].set_title(r'$\mathrm{Imag}\{\psi(t)\}$')
x_positions = np.arange(0, Nt, T*(Nt/tspan[1]))
x_labels = list(range(len(x_positions)))
for i in range(3):
    ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[i].set_xticks(x_positions)
    ax[i].set_xlabel('T')
    ax[i].set_xticklabels(x_labels)
    if i == 0:
        ax[i].set_ylabel('site')
    
fig.colorbar(plt.cm.ScalarMappable(cmap='Purples'), ax=ax, shrink=.5, pad=.01, aspect=10)
# fig.suptitle('F = '+str(a)+', omega='+str(omega)+ ', phi='+str(phi))
# fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#         'first_year_report/densityevolution,F=30,w=7p83,ph=0.pdf', 
#         format='pdf', bbox_inches='tight')
plt.show()



