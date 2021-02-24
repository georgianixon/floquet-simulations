
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
from hamiltonians import solve_schrodinger

import matplotlib as mpl
import seaborn as sns
from scipy.special import jv, jn_zeros

size=25
params = {
            'legend.fontsize': size,
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
mpl.rcParams.update(params)


def plotPsi(psi, x_positions, x_labels, title, normaliser):
    """
    Parameters
    ----------
    psi : TYPE
        Wavefunction to plot
    n_timesteps : TYPE
        number of timesteps generated between t0 and t_final
    n_oscillations : TYPE
        numer of full cycles
    title : TYPE
        title of graph

    Returns
    -------
    
    Graph

    """
    
    mpl.rcParams.update({
          'mathtext.fontset':'stix'
          })
    
    apply = [lambda x: np.abs(x)**2, np.real, np.imag]
    labels = [r'$|\psi(t)|^2$', r'$\mathrm{Re}\{\psi(t)\}$', r'$\mathrm{Imag}\{\psi(t)\}$']
    
    
    cmapcol = 'PuOr' #PiYG_r
    cmap= mpl.cm.get_cmap(cmapcol)
    
    sz = 6
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                            figsize=(sz*len(apply),sz))
    
    for i, f in enumerate(apply):
        ax[i].matshow(f(psi), interpolation='none', cmap=cmap, norm=normaliser, aspect='auto')
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
    
    cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmapcol, norm=normaliser), cax=cax)
    fig.suptitle(title, y = 1.11, fontsize=20, fontfamily='STIXGeneral')
    
    plt.show()


def phistring(phi):
    if phi == 0:
        return ""
    elif phi == "phi":
        return r'+ $\phi$' 
    else:
        return  r'$+ \pi /$' + str(int(1/(phi/pi)))
    
    
#%%


# choose particular HF

N = 71; A_site_start = 35;
centre = 25;
a = 35;
phi1=0; phi2=pi/2;
omega=a/jn_zeros(0,1)[0]
T=2*pi/omega

#when we solve scrodinger eq, how many timesteps do we want

n_oscillations = 60
n_timesteps = 60
n_osc_divisions = 4

tspan = (0,n_oscillations*T)

form = 'numericalG-SS-p'
rtol=1e-11

t_eval = np.linspace(tspan[0], tspan[1], n_timesteps)
psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;



psi_phi1 = solve_schrodinger(form, rtol, N, centre, a, None, None, omega, phi1, 
                                  tspan, n_timesteps, psi0)

psi_phi2 = solve_schrodinger(form, rtol, N, centre, a, None, None, omega, phi2 ,
                                  tspan, n_timesteps, psi0)
    


# normaliser= mpl.colors.Normalize(vmin=-1,vmax=1)
#%%
linthresh =1e-3
normaliser = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)

x_positions = np.linspace(0, n_timesteps, int(n_oscillations/n_osc_divisions+1))
x_labels = list(range(0, n_oscillations+1, n_osc_divisions))
    
title = ("Python; difference in "+r"$\psi$ for $\phi = 0$"
         + r" and $\phi = \pi / $" +str(int(1/(phi2/pi)))
         +  "\n" + r'$[ V(t) = '+str(a)+r'\cos( $' + str(round( omega, 2)) + r'$t$'
         + phistring("phi")
         + r'$) $'+' |25><25|]'  
         +', log scale, linthresh='+str(linthresh)
         +', rtol='+str(rtol)
        ) 
plotPsi(psi_phi1 - psi_phi2, x_positions, x_labels, title,
      normaliser)

title = ("Python; "
          +  "\n" + r'$[ V(t) = '+str(a)+r'\cos( $' + str(round( omega, 2)) + r'$t$'
          + phistring(phi1)
          + r'$) $'+' |25><25|]'  
          +', log scale, linthresh='+str(linthresh)
          +', rtol='+str(rtol)
         ) 

plotPsi(psi_phi1, x_positions, x_labels,  title,
      normaliser)

title = ("Python; "
          +  "\n" + r'$[ V(t) = '+str(a)+r'\cos( $' + str(round( omega, 2)) + r'$t$'
          + phistring(phi2)
          + r'$) $'+' |25><25|]'  
          +', log scale, linthresh='+str(linthresh)
          +', rtol='+str(rtol)
         ) 

plotPsi(psi_phi2, x_positions, x_labels,  title,
      normaliser)









#%%
# sz = 10
# fig, ax = plt.subplots(figsize=(sz,sz/2))
# ax.plot(np.linspace(0,N-1,N), (np.abs(psi_p.T[n_timesteps-1]))**2)
# ax.set_title(r'$V(t) = 35 \cos( $' + str(round( omega, 2)) + r'$t + \pi /$' +
#               str(int(1/(phi/pi))) + 
#               r'$) $'+' |25><25|' + '\n'+r'$G_{25,26} = $'+"{:.1g}".format(hopping) + '\n' +
#               r'$|\psi(t)|^2$ at $t_f$')
# ax.set_xlabel('n')
# plt.show()





    


        
        