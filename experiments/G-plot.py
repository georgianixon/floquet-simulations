# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:01:15 2020

@author: Georgia
"""

import matplotlib as mpl
place="Georgia Nixon"
from numpy import  cos, pi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
from floquet_simulations.hamiltonians import CreateHFGeneral

from scipy.linalg import eigh
from floquet_simulations.plot_functions import PlotParams, PlotAbsRealImagHamiltonian, PlotRealHamiltonian

PlotParams(fontsize=12)


posterLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Posters/202205 DAMOP/Final/"



"""
Plot HF---------------

Set form = ... and other parameters
Plot the Real, Imag and Abs parts of the floquet Hamiltonian
Using createHF general
"""


# to try and get black hole
# _, HF = CreateHFGeneral(3,
#                          [0,1,2],
#                          [Cosine]*3,
#                          [[,10,0,0] for i in range(10)],
#                          2*pi/10,
#                          0
#                          )
 
        
# common params
# N=8;   rtol=1e-11

# form="SS-p"
# phi1 = pi/4
# phiOffset = pi/2
# phi2 = phi1+phiOffset
# phis=[phi1, phi2];
# omega1 = a1/jn_zeros(0,1)[0]
# omegaMultiplier=2
# omega2 = omega1*omegaMultiplier
# omegas = [omega1, omega2]




#triangle
# N = 3
# centres= [1,2]
# funcs = [Cosine, Cosine]
# a1 = 15; a2 = 15;
# omega0 = 10
# T = 2*pi/omega0
# alpha = 1; beta = 2
# omega1 = alpha*omega0; omega2 = beta*omega0
# phi1 = 0; phi2 = 2*pi/3
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





# HFevals, HFevecs = GetEvalsAndEvecsGen(HF)
# HFabs = np.copy(HF)
# HFabs[0,2] = -np.abs(HFabs[0,2])
# HFabs[2,0] = -np.abs(HFabs[2,0])
# # HFabs = np.abs(HF)
# HFevalsabs, _ = GetEvalsAndEvecsGen(HFabs)


# sz = 7
# fig, ax = plt.subplots(figsize=(sz/2,sz))
# ax.plot([0]*len(HFevals), HFevals, 'o')
# ax.plot([1]*len(HFevals), HFevalsabs, 'o')
# # ax.axes.xaxis.set_visible(False)
# # ax.set_title("evals")
# ax.set_xticks([0, 1])
# ax.set_xticklabels([r'$\mathrm{evals}_{real}$', r'$\mathrm{evals}_{abs}$'])
# ax.set_xlim([-0.3, 1.4])
# plt.show()

#%%

a = 10
omega0 = 10
alpha = 1
beta = 3
phi2_frac = 0.8

tshift_frac = 0.1
phi3_frac = beta*(tshift_frac + phi2_frac/alpha)
phi3rel_frac = phi3_frac - phi2_frac

tshiftfrac1 = phi3rel_frac/beta + phi2_frac/beta  - phi2_frac/alpha
print(tshift_frac, tshiftfrac1)

t = np.linspace(0,2*pi/omega0,100)
two = 10*cos(alpha*omega0*t + phi2_frac*pi)
three = a*cos(beta*omega0*t + phi2_frac*pi + phi3rel_frac*pi)

plt.plot(t, two, label="two")
plt.plot(t, three, label="three")
plt.legend()
plt.show()



#%%


a = 10; omega = 2*pi; phi = 0; onsite = 0; theta = pi

t = np.linspace(0,2*pi/omega,1000)
y = RampGen([a, omega, phi, onsite, theta],t) 

plt.plot(t,y,'.')
#%%

""" Get A vals from specified tunnelling values
We use this for generating the linear gradient and the black hole stuff or also SSH using addition_type +2,-2"""

HF = np.real(HF)
#%%

def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags



#%%
 
# N = 5
# j1  = 0.9
# j2 = 0.5

# HF = np.diag(np.array([j1,j2]*(int(N-1))),-1)+np.diag(np.array([j1,j2]*(int(N-1))),1)

# Plot3G(HF)

HFevals, HFevecs = eigh(HF)


for i in range(2,7):#range(len(HFevecs)):
    fig, ax = plt.subplots()    
    ax.plot(range(len(HFevecs[:,0])), HFevecs[:,i])
    plt.show()

fig, ax = plt.subplots()
ax.plot(range(len(HFevals)), HFevals, 'x')
plt.show()



#%%

# def bmatrix(a):
#     """Returns a LaTeX bmatrix

#     :a: numpy array
#     :returns: LaTeX bmatrix as a string
#     """
#     if len(a.shape) > 2:
#         raise ValueError('bmatrix can at most display two dimensions')
#     lines = str(a).replace('[', '').replace(']\n', 'endline').replace("\n", "").replace("endline", "\n").replace("]]", "").splitlines()
#     rv = [r'\begin{bmatrix}']
#     rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
#     rv +=  [r'\end{bmatrix}']
#     return '\n'.join(rv)


# def tocsv(a):
#     """Returns a LaTeX bmatrix

#     :a: numpy array
#     :returns: LaTeX bmatrix as a string
#     """
#     if len(a.shape) > 2:
#         raise ValueError('bmatrix can at most display two dimensions')
#     lines = str(a).replace('[', '').replace(']\n', 'endline').replace("\n", "").replace("endline", "\n").replace("]]", "").splitlines()
#     rv = [r'\begin{bmatrix}']
#     rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
#     rv +=  [r'\end{bmatrix}']
#     return '\n'.join(rv)


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

HFdf = pd.DataFrame(np.real(HF))
HFdf.to_csv("/Users/GeorgiaNixon/OneDrive - Riverlane/PhD/LinearMetric/Hamiltonian_csvs_for_aydin/H_w=2_N=101_accumulativeA_withOnsites.csv", index=False, header=False)

# print(tocsv(signif(HF, 4)))
# print(bmatrix(signif(HF, 4)))
#%%

"""Get HF from general """
# _, HF = CreateHFGeneral(10,
#                           [0,1,2,3,4,5,6,7,8,9],
#                           [Cosine]*10,
#                           [[15,10,0,i/10] for i in range(10)], #a, omega, phi onsite
#                           2*pi/10,
#                           0
#                           )



