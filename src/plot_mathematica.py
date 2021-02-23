# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:13:18 2021

@author: Georgia
"""
import pandas as pd
import numpy as np
from numpy import pi
from fractions import Fraction
import matplotlib as mpl
import matplotlib.pyplot as plt

def convert_complex(s):
    return np.complex(s.replace('i', 'j').replace('*I', 'j').replace('*^', 'e'))


def plotPsi(psi, n_timesteps, n_oscillations, title, a, omega, phi):
    
    mpl.rcParams.update({
          'mathtext.fontset':'stix'
          })
    
    apply = [lambda x: np.abs(x)**2, np.real, np.imag]
    labels = [r'$|\psi(t)|^2$', r'$\mathrm{Re}\{\psi(t)\}$', r'$\mathrm{Imag}\{\psi(t)\}$']
    
    
    cmap= mpl.cm.get_cmap('PiYG_r')
    normaliser= mpl.colors.Normalize(vmin=-1,vmax=1)
    
    x_positions = np.linspace(0, n_timesteps, n_oscillations+1)
    x_labels = list(range(n_oscillations+1))
    
    sz = 17
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                            figsize=(sz,sz/2))
    
    for i, f in enumerate(apply):
        ax[i].matshow(f(psi), interpolation='none', cmap=cmap, norm=normaliser)
        ax[i].set_title(labels[i], fontsize=20, fontfamily='STIXGeneral')
        ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[i].set_xticks(x_positions)
        ax[i].set_xlabel('t/T', fontsize=15, fontfamily='STIXGeneral')
        ax[i].set_xticklabels(x_labels)
        for side in ["bottom", "top", "left", "right"]:
            ax[i].spines[side].set_visible(False)
        if i == 0:
            ax[i].set_ylabel('site', fontsize=15, fontfamily='STIXGeneral')
        
    fig.colorbar(plt.cm.ScalarMappable(cmap='PiYG_r', norm=normaliser), shrink=.5)
    fig.suptitle(title 
                 + "\n" + r'$V(t) = '+str(a)+r'\cos( $' + str(round( omega, 2)) + r'$t + \pi /$' +
                 str(int(1/(phi/pi))) + 
                  r'$) $'+' |25><25|'        
                  , y = 0.85, fontsize=30, fontfamily='STIXGeneral')
    
    plt.show()




sh = '/Users/Georgia/Code/MBQD/floquet-simulations/'
df = pd.read_csv(sh+'data/A35-w9p6-phpio7-mathematica-data.csv', 
                 index_col=False, 
                 header=None,
                 converters = dict.fromkeys(range(1000), convert_complex)
                )
assert(len(df.columns)<1000)

psi_m = df.to_numpy()

dfi = pd.read_csv(sh+'data/A35-w9p6-phpio7-mathematica-info.csv',
                  index_col = False)

N = dfi['Nlat'][0]; 
A_site_start = dfi['atom_start'][0];
centre = dfi['centre'][0]
a = dfi['A'][0]
phi=float(Fraction(dfi['phi/pi'][0]))*pi
omega = dfi['omega'][0]
T=2*pi/omega

n_oscillations = dfi['n_oscillations'][0]
n_timesteps = dfi['n_timesteps'][0]
tspan = (0,n_oscillations*T)
tstep = tspan[1]/ n_timesteps


#%%


plotPsi(psi_m, n_timesteps, n_oscillations, "Mathematica",
        a, omega, phi)


#%%

import sys
sys.path.append('/Users/Georgia/Code/MBQD/floquet-simulations/src')
from hamiltonians import solve_schrodinger

form = 'OSC'
t_eval = np.linspace(tspan[0], tspan[1], n_timesteps)
psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start-1] = 1;
rtol=1e-6

psi_p = solve_schrodinger(form, rtol, N, centre, a, None, None, omega, phi, 
                              tspan, n_timesteps, psi0)


plotPsi(psi_p, n_timesteps, n_oscillations, "Python",
        a, omega, phi)



#%%
#Compare


plotPsi(psi_p - psi_m, n_timesteps, n_oscillations, "Difference between Python and Mathematica",
        a, omega, phi)

#%%
















