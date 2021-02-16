
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:49 2020
|
@author: Georgia
"""


from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/Users/Georgia/Code/MBQD/floquet-simulations/src')
from hamiltonians import F_MG, F_OSC, create_HF, solve_schrodinger

import matplotlib
import seaborn as sns
from scipy.special import jv

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
phi=pi/7
omega=a*sin(pi/3)/pi
T=2*pi/omega

#when we solve scrodinger eq, how many timesteps do we want
n_timesteps = 100

# how many oscillations of hamiltonian do we want to calculate psi for?
n_oscillations = 15

# define beginnning and end times to solve for
tspan = (0,n_oscillations*T)

form = 'theoretical_hermitian'
rtol=1e-6

t_eval = np.linspace(tspan[0], tspan[1], n_timesteps)
psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;

hopping = exp(1j*a*sin(phi)/omega)*jv(0,a/omega)

import numpy as np
import matplotlib as mpl

cmap= mpl.cm.get_cmap('Purples')
normaliser= mpl.colors.Normalize(vmin=0,vmax=1)

if form=='theoretical' or form == 'theoretical_hermitian':
    psievolve = solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi, 
                                  tspan, n_timesteps, psi0)
    
else:
    sol = solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi, 
                            tspan, n_timesteps, psi0)
    psievolve = sol.y
    

sz = 10


fig, ax = plt.subplots(figsize=(sz,sz/2))
ax.plot(np.linspace(0,N-1,N), (np.abs(psievolve.T[n_timesteps-1]))**2)
ax.set_title(r'$V(t) = 35 \cos( $' + str(round( omega, 2)) + r'$t + \pi /$' +
              str(int(1/(phi/pi))) + 
              r'$) $'+' |25><25|' + '\n'+r'$G_{25,26} = $'+"{:.1g}".format(hopping) + '\n' +
              r'$|\psi(t)|^2$ at $t_f$')
ax.set_xlabel('n')
plt.show()


sz=15
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                        figsize=(sz,sz/2))
ax[0].matshow(abs(psievolve)**2, interpolation='none', cmap=cmap, norm=normaliser)
ax[1].matshow(np.real(psievolve), interpolation='none', cmap=cmap, norm=normaliser)
ax[2].matshow(np.imag(psievolve), interpolation='none', cmap=cmap, norm=normaliser)
ax[0].set_title(r'$|\psi(t)|^2$')
ax[1].set_title(r'$\mathrm{Re}\{\psi(t)\}$')
ax[2].set_title(r'$\mathrm{Imag}\{\psi(t)\}$')
x_positions = np.linspace(0, n_timesteps, n_oscillations+1)
x_labels = list(range(n_oscillations+1))
for i in range(3):
    ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[i].set_xticks(x_positions)
    ax[i].set_xlabel('T')
    ax[i].set_xticklabels(x_labels)
    if i == 0:
        ax[i].set_ylabel('site')



fig.colorbar(plt.cm.ScalarMappable(cmap='Purples'), shrink=.5)

fig.suptitle(r'$V(t) = 35 \cos( $' + str(round( omega, 2)) + r'$t + \pi /$' +
             str(int(1/(phi/pi))) + 
              r'$) $'+' |25><25|' + '\n'+r'$G_{25,26} = $'+"{:.1g}".format(hopping) + 
              r', $\quad |G_{25,26}| = $'+ "{:.1f}".format(np.abs(hopping)), y = 0.85)

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
omega=7; 
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
        
        
        
        