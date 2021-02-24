# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:01:15 2020

@author: Georgia
"""

from numpy.linalg import eig
import matplotlib.colors as col

from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import sys
sys.path.append('/Users/Georgia/Code/MBQD/floquet-simulations/src')
from hamiltonians import create_HF

from scipy.special import jn_zeros


import matplotlib

size=20
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
matplotlib.rcParams.update(params)


#%%

"""
Plot HF---------------

Set form = ... and other parameters
Plot the Real, Imag and Abs parts of the floquet Hamiltonian
"""


N=51; centre=25; a=35; phi=pi/2;
omega=a/jn_zeros(0,1)[0]
# omega=9.6
# form='SS-p'
form='linear'
rtol=1e-11
UT, HF = create_HF(form, rtol, N, centre, a,None, None,phi, omega)

#%%

norm = col.Normalize(vmin=-1, vmax=1)
linthresh = 1e-8
norm=col.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)


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
    ax[n1].set_title(labels[n1], fontsize=25)
    ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[n1].set_xlabel('m', fontsize=20)

ax[0].set_ylabel('n', rotation=0, labelpad=10, fontsize=20)

    
    
cax = plt.axes([1.03, 0.1, 0.03, 0.8])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
# fig.colorbar(pcm, ax=ax[0], extend='max')

fig.suptitle('Python'
             +', Linear'
    + r', $V_{n,n}(t) = $'+
    # str(a)+r'$ \cos( \omega t)$'
       str(a)+r'$n \cos( $'
        # +str(omega)
        +"{:.2f}".format(omega)
       +r'$ t$'
        + r'$ + \pi /$' + str(int(1/(phi/pi))) 
       + ')'
        +'\n'+'linthresh='+str(linthresh)
        +', rtol='+str(rtol)
    , fontsize = 25, y=0.96)


#             
#fig.savefig('', 
#        format='pdf', bbox_inches='tight')
plt.show()







