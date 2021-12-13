# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:01:15 2020

@author: Georgia
"""

import matplotlib as mpl
place="Georgia"
from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
# sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
from hamiltonians import CreateHF, CreateHFGeneral, HT_SS, hoppingHF, GetEvalsAndEvecsGen
from scipy.special import jn_zeros, jv
from fractions import Fraction 
from hamiltonians import Cosine, PhiString
    
def RemoveWannierGauge(matrix, c, N):
    phase = np.angle(matrix[c-1,c])
    gaugeMatrix = np.identity(N, dtype=np.complex128)
    gaugeMatrix[c,c] = np.exp(-1j*phase)
    matrix = np.matmul(np.matmul(np.conj(gaugeMatrix), matrix), gaugeMatrix)
    return matrix


size=12
params = {
            'legend.fontsize': size*0.8,
#          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'font.size': size,
          'xtick.bottom':True,
          'xtick.top':False,
          'ytick.left': True,
          'ytick.right':False,
          ## draw ticks on the left side
#          'axes.titlepad': 25
          'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
          'axes.grid':False,
          'font.family' : 'STIXGeneral',
          'mathtext.fontset':'stix'
          }
mpl.rcParams.update(params)




#%%

"""
Plot HF---------------

Set form = ... and other parameters
Plot the Real, Imag and Abs parts of the floquet Hamiltonian
"""

# common params
N=12;   rtol=1e-11

# form="SS-p"
# phi1 = pi/4
# phiOffset = pi/2
# phi2 = phi1+phiOffset
# phis=[phi1, phi2];
# omega1 = a1/jn_zeros(0,1)[0]
# omegaMultiplier=2
# omega2 = omega1*omegaMultiplier
# omegas = [omega1, omega2]


form="SSHModel"#"StepFunc"#"SS-p"
centres= [1]
centre = 0
a = [35,35]
omega = [32, 7]#8.1
# T = 2*pi/omega
phi = [0,0]
onsite = [0,0]
funcs = [Cosine]
paramss = [[a, omega, phi, onsite]]
circleBoundary = 1

UT, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)
# UT, HF = CreateHFGeneral(N, centres, funcs, paramss, T, circleBoundary)
# HFevals, HFevecs = GetEvalsAndEvecsGen(HF)


#%%

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# linthresh = 1e-1
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
# 

'''abs real imag'''

apply = [
         np.abs, 
         np.real, np.imag]
labels = [
          r'$\mathrm{Abs}\{G_{n,m}\}$', 
          r'$\mathrm{Re}\{G_{n,m}\}$',
          r'$\mathrm{Imag}\{G_{n,m}\}$'
          ]

sz = 6
fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))

for n1, f in enumerate(apply):
    pcm = ax[n1].matshow(f(HF), interpolation='none', cmap='PuOr',  norm=norm)
    ax[n1].set_title(labels[n1])
    ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[n1].set_xlabel('m')

ax[0].set_ylabel('n', rotation=0, labelpad=10)

    
    
cax = plt.axes([1.03, 0.1, 0.03, 0.8])
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)

# fig.suptitle('Python'
#                # +', SS'
#                +", Hopping toy model"
#              # +', Linear'
#     + r', $V(t) = $'
#     + r"$|25><25|$"
#     # str(a)+r'$ \cos( \omega t)$'
#     + str(a)+r'$\cos( $'
#     # +str(omega)
#     + "{:.2f}".format(omega)
#     + r'$ t$'
#     + phistring(phi)
#     + ')'
#     + '\n'+'linthresh='+str(linthresh)
#     + ', rtol='+str(rtol)
#     , fontsize = 25, y=0.96)

# fig.suptitle("Representation of Floquet Hamiltonian, G\n"
#              # + r"given $H(t)=H_0 + 35 \cos (" + "{:.2f}".format(omega1)
#              # + r"t" + phistring(phi1) 
#              # + r") |"+str(centre)+r"><"+str(centre) +r"|$",
#              +form+", "+r"$a_1=$"+str(a1)
#               +", "+r"$a_2=$"+str(a2)
#              +", "+r"$\omega_1=$"+"{:.2f}".format(omega1)
#               +", "+r"$\omega_2=$"+"{:.2f}".format(omega2)
#               +", "+r"$\phi_1=$"+ PhiStringNum(phi1)
#               +", "+r"$\phi_2=$"+ PhiStringNum(phi2)
             # , y=0.95)
             
#     + r', $V(t) = $'
#     + r"$|25><25|$"
#     # str(a)+r'$ \cos( \omega t)$'
#     + str(a)+r'$\cos( $'
#     # +str(omega)
#     + "{:.2f}".format(omega)
#     + r'$ t$'
#     + phistring(phi)
#     + ')'
#     + '\n'+'linthresh='+str(linthresh)
#     + ', rtol='+str(rtol)
#     , fontsize = 25, y=0.96)


#             
#fig.savefig('', 
#        format='pdf', bbox_inches='tight')
plt.show()

#%%

# fig for paper


norm = mpl.colors.Normalize(vmin=0, vmax=1)
# linthresh = 1e-1
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
cmapcol = "Purples"#'PuOr' #PiYG_r
cmap= mpl.cm.get_cmap(cmapcol)


'''abs real imag'''

apply = [np.abs]
labels = [r'$\mathrm{Abs}\{G_{n,m}\}$']

sz = 2.1
fig, ax = plt.subplots( constrained_layout=True, figsize=(sz,sz))


pcm = ax.matshow(np.abs(HF), interpolation='none', cmap='Purples',  norm=norm)
ax.set_title(labels[0])
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
  labeltop=False)  
ax.set_xlabel('m')

ax.set_ylabel('n', rotation=0, labelpad=10)

    
    
cax = plt.axes([0.92, 0.12, 0.06, 0.8])
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)



paper = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Paper/Figures/"
fig.savefig(paper+'G-SSHModel-small.pdf', format='pdf', bbox_inches='tight')
plt.show()

#%%

# for poster - real part of Hamiltonian Triangle

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# linthresh = 1e-1
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
# 

'''abs real imag'''


sz = 1.8
fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, 
                       figsize=(sz,sz))

pcm = ax.matshow(np.real(HF), interpolation='none', cmap='PuOr',  norm=norm)
ax.set_title(r'$\mathrm{Real}\{G_{n,m}\}$')
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
  labeltop=False)  
ax.set_xlabel('m')

ax.set_ylabel('n', rotation=0, labelpad=10)

ax.set_ylabel('n', rotation=0, labelpad=10)
    
cax = plt.axes([0.92, 0.12, 0.08, 0.8])
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)
fig.savefig(paper+'G-Triangle-Real.pdf', format='pdf', bbox_inches='tight')
plt.show()






