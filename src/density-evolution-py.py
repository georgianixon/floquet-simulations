
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:49 2020
|
@author: Georgia
"""

place = "Georgia Nixon"
from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import SolveSchrodinger

import matplotlib as mpl
import seaborn as sns
from scipy.special import jv, jn_zeros
from fractions import Fraction

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/"

size=25
params = {
            'legend.fontsize': size*0.75,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size,
          'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
          'axes.grid':False,
          'font.family' : 'STIXGeneral',
          'mathtext.fontset':'stix'
          }
mpl.rcParams.update(params)


def PlotPsi(psi, x_positions, x_labels, title, normaliser):
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
    
    sz = 5
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                            figsize=(sz*len(apply),sz))
    
    for i, f in enumerate(apply):
        ax[i].matshow(f(psi), interpolation='none', cmap=cmap, norm=normaliser, aspect='auto')
        ax[i].set_title(labels[i],  fontfamily='STIXGeneral')
        ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[i].set_xticks(x_positions)
        ax[i].set_xlabel('t/T', fontfamily='STIXGeneral')
        ax[i].set_xticklabels(x_labels)
        for side in ["bottom", "top", "left", "right"]:
            ax[i].spines[side].set_visible(False)
        if i == 0:
            ax[i].set_ylabel('site', fontfamily='STIXGeneral')
    
    cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmapcol, norm=normaliser), cax=cax)
    fig.suptitle(title, y = 1.2,  fontfamily='STIXGeneral')
    
    plt.show()
    
def PlotTwoPsi(psi1, psi2, x_positions, x_labels, title, normaliser):
    """
    Plot the difference between two wavefunctison
    |psi1|^2 - |psi2|^2
    Re[psi1 - psi2]
    Im[psi1 - psi2]

    """
    
    apply = [lambda x,y: np.abs(x)**2 - np.abs(y)**2, 
             lambda x,y: np.abs(x - y)**2, 
             lambda x,y: np.real(x - y),
             lambda x,y: np.imag(x - y)]
    labels = [r'$|\psi_1(t)|^2 - |\psi_2(t)|^2$', 
              r'$|\psi_1(t) - \psi_2(t)|^2$',
              r'$\mathrm{Re}\{\psi_1(t) - \psi_2(t)\}$', r'$\mathrm{Imag}\{\psi_1(t) - \psi_2(t)\}$']
    
    
    cmapcol = 'PuOr' #PiYG_r
    cmap= mpl.cm.get_cmap(cmapcol)
    
    sz = 5
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                            figsize=(sz*len(apply),sz))
    
    for i, f in enumerate(apply):
        ax[i].matshow(f(psi1, psi2), interpolation='none', cmap=cmap, norm=normaliser, aspect='auto')
        ax[i].set_title(labels[i])
        ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[i].set_xticks(x_positions)
        ax[i].set_xlabel('t/T')
        ax[i].set_xticklabels(x_labels)
        for side in ["bottom", "top", "left", "right"]:
            ax[i].spines[side].set_visible(False)
        if i == 0:
            ax[i].set_ylabel('site')
    
    cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmapcol, norm=normaliser), cax=cax)
    fig.suptitle(title, y = 1.2)
    plt.show()
    


def PhiString(phi):
    if phi == 0:
        return ""
    elif phi == "phi":
        return r'+ \phi' 
    else:
        return  r'+ \pi /' + str(int(1/(phi/pi)))
    
def phistringnum(phi):
    if phi == 0:
        return "0"
    elif phi == "phi":
        return r'\phi' 
    else:
        return  r'\pi /' + str(int(1/(phi/pi)))
    
    
#%%


# choose particular HF

N = 91; A_site_start = 40;
centre = 40;
a = 35;
phi1=pi/2;
phi2=0;
omega= a /jn_zeros(0,1)[0]
# omega = 10
T=2*pi/omega

#when we solve scrodinger eq, how many timesteps do we want

nOscillations = 30
#how many steps we want. NB, this means we will solve for nTimesteps+1 times 
nTimesteps = nOscillations
n_osc_divisions = 2

tspan = (0,nOscillations*T)

# form1 = 'SS-p'
form2 = 'numericalG-SS-p'
rtol=1e-11

t_eval = np.linspace(tspan[0], tspan[1], nTimesteps)
psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;


psi1 = SolveSchrodinger(form2, rtol, N, centre, a, omega, phi1, 
                                  tspan, nTimesteps, psi0)

psi2 = SolveSchrodinger(form2, rtol, N, centre, a,omega, phi2,
                                  tspan, nTimesteps, psi0)
    


# normaliser= mpl.colors.Normalize(vmin=-1,vmax=1)
#%%

    
# normaliser = mpl.colors.Normalize(vmin=-1, vmax=1)
linthresh = 1e-3
normaliser=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)

x_positions = np.linspace(0, nTimesteps, int(nOscillations/n_osc_divisions+1))
x_labels = list(range(0, nOscillations+1, n_osc_divisions))

title = (r"$|\psi_1> = |\psi_{\phi="+phistringnum(phi1)+"}(t)>, \>"
         +"|\psi_2> = |\psi_{\phi="+phistringnum(phi2)+"}(t)>$"
         +"\n"
         +r"evolution via G, "
         + r"given $H(t)=H_0 + "+str(a)
         +r"\cos (" + "{:.2f}".format(omega)+ r"t + \phi"
         + r") |"+str(centre)+r"><"+str(centre) +r"|,"
         +r" \quad  |\psi (t=0)> = |"+str(A_site_start)+r">$"
        )  

PlotTwoPsi(psi1, psi2, x_positions, x_labels, title,
      normaliser)


title = (r"$|\psi (t)>$"+  "\n"
         +r"evolution via G, "
         + r"given $H(t)=H_0 + 35 \cos (" + "{:.2f}".format(omega)
             + r"t" + PhiString(phi1) 
             + r") |"+str(centre)+r"><"+str(centre) +r"|,"
             +r" \quad  |\psi (t=0)> = |"+str(A_site_start)+r">$"
         ) 

PlotPsi(psi1, x_positions, x_labels,  title,
      normaliser)


title = (r"$|\psi (t)>$"+  "\n"
         +r"evolution via G, "
         + r"given $H(t)=H_0 + 35 \cos (" + "{:.2f}".format(omega)
             + r"t" + PhiString(phi2) 
             + r") |"+str(centre)+r"><"+str(centre) +r"|,"
             +r" \quad  |\psi (t=0)> = |"+str(A_site_start)+r">$"
         ) 

PlotPsi(psi2, x_positions, x_labels,  title,
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





    


        
        