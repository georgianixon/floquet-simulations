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
sys.path.append('/Users/Georgia/Code/MBQD/lattice-simulations')
from hamiltonians import create_HF


import matplotlib

size=16
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


N=51; centre=25; a=35; phi=0; omega=6.34
form='OSC'
UT, HF = create_HF(form, N, centre, a,np.nan, np.nan,phi, omega)

'''One large'''
sz = 10
fig, ax = plt.subplots(figsize=(sz,sz))
norm = col.Normalize(vmin=-1, vmax=1)
ax.matshow(np.imag(HF), interpolation='none', cmap='PuOr', norm=norm)
ax.tick_params(axis="x", bottom=True, top=False,  labelbottom=True,  labeltop=False)
ax.set_xlabel('m')
ax.set_ylabel('n', rotation=0, labelpad=10)

cax = plt.axes([1, 0.05, 0.06, 0.9])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
# fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#         'first_year_report/HF,F=30,w=8,ph=0.pdf', 
#         format='pdf', bbox_inches='tight')
plt.show()

'''abs real imag'''
sz = 20
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))
ax[0].matshow(np.abs(HF), interpolation='none', cmap='PuOr',  norm=norm)
ax[1].matshow(np.real(HF), interpolation='none', cmap='PuOr',  norm=norm)
ax[2].matshow(np.imag(HF), interpolation='none', cmap='PuOr',  norm=norm)
ax[0].set_title(r'$\mathrm{Abs}\{G_{n,m}\}$')
ax[1].set_title(r'$\mathrm{Re}\{G_{n,m}\}$')
ax[2].set_title(r'$\mathrm{Imag}\{G_{n,m}\}$')


ax[0].set_ylabel('n', rotation=0, labelpad=10)
for i in range(3):
    ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[i].set_xlabel('m')
    
norm = col.Normalize(vmin=-1, vmax=1) 
cax = plt.axes([1.03, 0.1, 0.03, 0.8])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
#             
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#        'first_year_report/HF,MGSTA,a=30,b=1,c=1,w=5,ph=0.pdf', 
#        format='pdf', bbox_inches='tight')
plt.show()

