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
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import CreateHF, HT_SS, hoppingHF
from scipy.special import jn_zeros, jv
from fractions import Fraction 
    
    
def PhiString(phi):
    fraction = phi/pi
    fraction = Fraction(fraction).limit_denominator(100)
    numerator = fraction.numerator
    denominator = fraction.denominator
    if numerator ==1:
        return r"\pi /"+str(denominator)
    else:
        str(numerator)+r"\pi / "+str(denominator)

size=25
params = {
            'legend.fontsize': size*0.75,
#          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
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
N=51;   rtol=1e-11
a = 35

# form="SS-p"
# phi1 = pi/4
# phiOffset = pi/2
# phi2 = phi1+phiOffset
# phis=[phi1, phi2];
# omega1 = a1/jn_zeros(0,1)[0]
# omegaMultiplier=2
# omega2 = omega1*omegaMultiplier
# omegas = [omega1, omega2]


form="StepFuncGen"
centre=[ 25]
a = [ 35]
omega = [10]
phi = [ 0]
onsite = [0]


UT, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)

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

sz = 20
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
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
# fig.colorbar(pcm, ax=ax[0], extend='max')

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







