# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:01:15 2020

@author: Georgia
"""

import matplotlib as mpl
place="Georgia Nixon"
from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
from hamiltonians import CreateHF, CreateHFGeneral, HT_SS, hoppingHF, GetEvalsAndEvecsGen
from hamiltonians import Cosine, RemoveWannierGauge
from scipy.special import jn_zeros, jv
from fractions import Fraction 
from hamiltonians import Cosine, PhiString
    
posterLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Posters/202205 DAMOP/Final/"
# def RemoveWannierGauge(matrix, c, N):
#     phase = np.angle(matrix[c-1,c])
#     gaugeMatrix = np.identity(N, dtype=np.complex128)
#     gaugeMatrix[c,c] = np.exp(-1j*phase)
#     matrix = np.matmul(np.matmul(np.conj(gaugeMatrix), matrix), gaugeMatrix)
#     return matrix


def Plot():
    size=22
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
              'font.family' : "sans-serif",#"Arial"#'STIXGeneral',
              "font.sans-serif":"Arial",
               'mathtext.fontset':"Arial"#'stix'
              }
    

Plot()



#%%

"""
Plot HF---------------

Set form = ... and other parameters
Plot the Real, Imag and Abs parts of the floquet Hamiltonian
"""

# def GetPhiOffsetReal(phi1Real, phiTOffset, omegaMultiplier):
#     if omegaMultiplier == 1.5:
#         piEquivA = 1/4
#         piEquivB = 1/6
#     phi1T = phi1Real/pi*piEquivA
#     phi2T = phi1T+phiTOffset
#     phi2Real = phi2T/piEquivB*pi
#     return phi2Real

def GetPhiOffset(time1, timeOffset, omega1, omega2):
    time2 = time1+timeOffset
    
    omegaT = np.gcd(round(100*omega1), round(100*omega2))/100
    totalT = 2*pi/omegaT
    
    phi1 = time1*omega1*totalT
    phi2 = time2*omega2*totalT
    
    return phi1, phi2, totalT 
        
# common params
N=8;   rtol=1e-11

# form="SS-p"
# phi1 = pi/4
# phiOffset = pi/2
# phi2 = phi1+phiOffset
# phis=[phi1, phi2];
# omega1 = a1/jn_zeros(0,1)[0]
# omegaMultiplier=2
# omega2 = omega1*omegaMultiplier
# omegas = [omega1, omega2]


form = "SSHModel"
centre = np.nan
a = [35, 15]
omega1 = 10; omega = [omega1, 2*omega1 ]
phi = [0, 0]
onsite = [0,0]
UT, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)

#triangle
# centres= [1,2]
# funcs = [Cosine, Cosine]
# a1 = 25; a2 = 53;
# omega0 = 6
# T = 2*pi/omega0
# alpha = 1; beta = 2
# omega1 = alpha*omega0; omega2 = beta*omega0
# phi1 = 0; phi2 = pi/3
# onsite = 0
# paramss = [[a1, omega1, phi1, onsite], [a2, omega2, phi2, onsite]]
# circleBoundary = 1


# a = 35
# omegaMultiplier = 1.5
# omega1 = 6.8; omega2=omegaMultiplier*omega1
# time1 = 4/8
# timeOffset  = 2/3 # units of total time period
# phi1, phi2, T  = GetPhiOffset(time1, timeOffset, omega1, omega2)



#full lattice shake
# form = 'linear'
# centre = np.nan
# a = 10
# omega = 10
# phi = pi/3
# onsite = np.nan
# _, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)


# _, HF = CreateHFGeneral(N, centres, funcs, paramss, T, circleBoundary)


# for site in range(N):
#     HF = RemoveWannierGauge(HF, site, N)




# HFevals, _ = GetEvalsAndEvecsGen(HF)
# HFabs = np.copy(HF)
# HFabs[0,2] = -np.abs(HFabs[0,2])
# HFabs[2,0] = -np.abs(HFabs[2,0])
# # HFabs = np.abs(HF)
# HFevalsabs, _ = GetEvalsAndEvecsGen(HFabs)


# sz = 7
# fig, ax = plt.subplots(figsize=(sz/2,sz))
# ax.plot([0,0,0], HFevals, 'o')
# ax.plot([1,1,1], HFevalsabs, 'o')
# # ax.axes.xaxis.set_visible(False)
# # ax.set_title("evals")
# ax.set_xticks([0, 1])
# ax.set_xticklabels([r'$\mathrm{evals}_{real}$', r'$\mathrm{evals}_{abs}$'])
# ax.set_xlim([-0.3, 1.4])
# plt.show()


#%%

latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Analytics/"
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

sz = 8
fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))

for n1, f in enumerate(apply):
    pcm = ax[n1].matshow(f(HF), interpolation='none', cmap='PuOr',  norm=norm)
    ax[n1].set_title(labels[n1])
    ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[n1].set_xlabel('m')

ax[0].set_ylabel('n', rotation=0, labelpad=10)

# cax = plt.axes([1.03, 0.1, 0.03, 0.8])
cax = plt.axes([1.03, 0.2, 0.03, 0.6])
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)
    
latexLoc = "C:/Users/Georgia/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Analytics/"
# fig.savefig(latexLoc+"G,Triangle,Example,Gauged.pdf", 
#         format='pdf', bbox_inches='tight')
plt.show()




#%%



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
# fig.savefig(paper+'G-SSHModel-small.pdf', format='pdf', bbox_inches='tight')
plt.show()

#%%

#  Real part of Hamiltonian 

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# linthresh = 1e-2
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
# 

'''abs real imag'''
Plot()
cmap = "RdBu"#"PuOr"

sz = 2
fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, 
                       figsize=(sz,sz))

pcm = ax.matshow(np.real(HF), interpolation='none', cmap=cmap,  norm=norm)
ax.set_title(r'$\mathrm{Real}\{H_{n,m}^{\mathrm{eff}}\}$')
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
  labeltop=False)  
ax.set_xlabel('m')

ax.set_ylabel('n', rotation=0, labelpad=10)

ax.set_ylabel('n', rotation=0, labelpad=14)
    
# cax = plt.axes([0.97, 0.145, 0.08, 0.78]) #size = 6
# cax = plt.axes([0.99, 0.13, 0.04, 0.8])
cax = plt.axes([0.9, 0.265, 0.08, 0.59]) 
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)
fig.savefig(posterLoc+'G-SSH-Real.pdf', format='pdf', bbox_inches='tight')
plt.show()

#%%

# Abs part of Hamiltonian 

norm = mpl.colors.Normalize(vmin=0, vmax=1)
# linthresh = 1e-1
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
# 

'''abs real imag'''



sz = 2
fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, 
                       figsize=(sz,sz))

pcm = ax.matshow(np.abs(HF), interpolation='none', cmap='Purples',  norm=norm)
ax.set_title(r'$\mathrm{Abs}\{G_{n,m}\}$')
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
  labeltop=False)  
ax.set_xlabel('m')

ax.set_ylabel('n', rotation=0, labelpad=10)

ax.set_ylabel('n', rotation=0, labelpad=14)
    
cax = plt.axes([0.97, 0.145, 0.08, 0.78])
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)
# fig.savefig(paper+'G-Triangle-Real.pdf', format='pdf', bbox_inches='tight')
plt.show()





