# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:49 2020

@author: Georgia
"""


from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/Users/Georgia/Code/MBQD/floquet-simulations')
from hamiltonians import F_MG, F_OSC, create_HF, solve_schrodinger

import matplotlib
import seaborn as sns

from math import log as mathlog


size=25
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

N = 51; A_site_start = 35;
centre = 25
a = 35;
b = np.nan
c = np.nan 
omega=6.34; 
phi=pi/2;
T=2*pi/omega
tspan = (0,10)
Nt = 100
form = 'theoretical'
rtol=1e-7

t_eval = np.linspace(tspan[0], tspan[1], Nt)
psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;



import numpy as np
import matplotlib as mpl

cmap= mpl.cm.get_cmap('Purples')
normaliser= mpl.colors.Normalize(vmin=0,vmax=1)

sol = solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi, tspan, psi0, avg=1)

sz = 10


fig, ax = plt.subplots(figsize=(sz,sz/2))
ax.plot(t_eval,(np.abs(sol.y)[25])**2)
ax.set_title('occupation of centre site')
ax.set_xlabel('t')
plt.show()


sz=20
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                        figsize=(sz,sz/2))
ax[0].matshow(abs(sol.y)**2, interpolation='none', cmap=cmap, norm=normaliser)
ax[1].matshow(np.real(sol.y), interpolation='none', cmap=cmap, norm=normaliser)
ax[2].matshow(np.imag(sol.y), interpolation='none', cmap=cmap, norm=normaliser)
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


fig.colorbar(plt.cm.ScalarMappable(cmap='Purples'), shrink=.5, pad=.05, aspect=15, fraction=.1)
# fig.suptitle('F = '+str(a)+', omega='+str(omega)+ ', phi='+str(phi))
# fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#         'first_year_report/densityevolution,F=30,w=7p83,ph=0.pdf', 
#         format='pdf', bbox_inches='tight')
plt.show()

#%%


# plot only psi squared

N = 51; A_site_start = 25;
centre = 20
a = 35;
b = np.nan
c = np.nan 
omega=6.34; 
phi=0;
T=2*pi/omega
tspan = (0,10)
Nt = 100
form = 'OSC'
rtol=1e-7

cmap_name = 'Blues'

t_eval = np.linspace(tspan[0], tspan[1], Nt)
psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;


import numpy as np
import matplotlib as mpl

cmap= mpl.cm.get_cmap(cmap_name)
normaliser= mpl.colors.Normalize(vmin=0,vmax=1)
# normaliser= mpl.colors.LogNorm(vmin=0,vmax=1)

sol = solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi, tspan, psi0, avg=1)

sz=10
fig, ax = plt.subplots(figsize=(sz,sz/2))
ax.matshow(abs(sol.y)**2, interpolation='none', cmap=cmap, norm=normaliser)
ax.set_title(r'$|\psi(t)|^2$')
x_positions = np.arange(0, Nt, T*(Nt/tspan[1]))
x_labels = list(range(len(x_positions)))
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
ax.set_xticks(x_positions)
ax.set_xlabel('T')
ax.set_xticklabels(x_labels)
ax.set_ylabel('site')


fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_name))
# fig.colorbar(plt.cm.ScalarMappable(cmap='Purples'), shrink=.9, pad=.05, aspect=15, fraction=.1)
# fig.suptitle('F = '+str(a)+', omega='+str(omega)+ ', phi='+str(phi))
# fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#         'first_year_report/densityevolution,F=30,w=7p83,ph=0.pdf', 
#         format='pdf', bbox_inches='tight')
plt.show()
        
        
        
        