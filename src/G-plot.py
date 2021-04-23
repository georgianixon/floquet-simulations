# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:01:15 2020

@author: Georgia
"""

from numpy.linalg import eig
import matplotlib as mpl

from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/Users/Georgia/Code/MBQD/floquet-simulations/src')
from hamiltonians import create_HF, HT_SS, hoppingHF

from scipy.special import jn_zeros, jv

def phistring(phi):
    if phi == 0:
        return ""
    elif phi == "phi":
        return r'+ \phi' 
    else:
        return  r'+ \pi /' + str(int(1/(phi/pi)))

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


N=51; centre=25; a=35; phi=pi/2;
omega=a/jn_zeros(0,1)[0]
omega=9.6
form='SS-p'
rtol=1e-20
UT, HF = create_HF(form, rtol, N, centre, a,phi, omega)

# HF, entry = hoppingHF(N, centre, a, omega, phi)

# HF = HT_SS(N, centre, a, omega, 1, phi)
#%%


norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# linthresh = 1e-3
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)


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

fig.suptitle("Representation of Floquet Hamiltonian, G\n"
             + r"given $H(t)=H_0 + 35 \cos (" + "{:.2f}".format(omega)
             + r"t" + phistring(phi) 
             + r") |"+str(centre)+r"><"+str(centre) +r"|$",
             y=0.95)
             
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

diff = 5
a = roundcomplex(np.sum([HF[i,i] for i in range(25-diff, 25+diff+1)]), 9)
b = roundcomplex(np.sum([HF[i,i-1] for i in range(26-diff, 26+diff)]), 9)
c= roundcomplex(np.sum([HF[i,i-2] for i in range(26-diff, 26+diff+1)]), 9)
d = roundcomplex(np.sum([HF[i,i-3] for i in range(27-diff, 27+diff)]), 9)
e = roundcomplex(np.sum([HF[i,i-4] for i in range(27-diff, 27+diff+1)]), 9)
f = roundcomplex(np.sum([HF[i,i-5] for i in range(28-diff, 28+diff)]), 9)
g = roundcomplex(np.sum([HF[i,i-6] for i in range(28-diff, 28+diff+1)]), 9)
h = roundcomplex(np.sum([HF[i,i-7] for i in range(29-diff, 29+diff)]), 9)
i = roundcomplex(np.sum([HF[i,i-8] for i in range(29-diff, 29+diff+1)]), 9)


print(a+b+c+d+e+f+g+h+i)

# lowertriangle = 




