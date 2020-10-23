# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:49 2020

@author: Georgia
"""


from numpy import exp, sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.special import jv, jn_zeros
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%

N = 51; A_site_start = 25;
a = 30; 
b=1; c=1;
omega=7.83; phi=pi/7; T=2*pi/omega
centre = 25;
tspan = (0,10)
Nt = 100
t_eval = np.linspace(tspan[0], tspan[1], Nt)

psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;

sol = solve_ivp(lambda t,psi: F_OSC(t, psi, N, centre,
                                                     a,
                                                     omega, phi), 
                                    tspan, psi0, rtol=1e-7, atol=1e-7,
                                    t_eval=t_eval, method='RK45')



print('phi = '+str(phi))
print('omega = '+str(omega))

#sz = 7
sz = 12
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))
ax[0].matshow(abs(sol.y)**2, interpolation='none', cmap='Purples')
ax[1].matshow(np.real(sol.y), interpolation='none', cmap='Purples')
ax[2].matshow(np.angle(sol.y), interpolation='none', cmap='Purples')
#ax[3].matshow(np.angle(HF), interpolation='none', cmap='PuOr', norm=norm_phase)
ax[0].set_title(r'$|\psi(t)|^2$')
ax[1].set_title(r'$\mathrm{Re}\{\psi(t)\}$')
ax[2].set_title(r'$\mathrm{Imag}\{\psi(t)\}$')
#ax[3].set_title('phase')
x_positions = np.arange(0, Nt, T*(Nt/tspan[1]))
x_labels = list(range(len(x_positions)))
for i in range(3):
    ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[i].set_xticks(x_positions)
    ax[i].set_xlabel('T')
    ax[i].set_xticklabels(x_labels)
    if i == 0:
        ax[i].set_ylabel('site')
    
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)

#fig.colorbar(plt.cm.ScalarMappable(cmap='Purples'), ax=ax, shrink=.5, pad=.01, aspect=10)
#fig.suptitle('F = '+str(a)+', omega='+str(omega)+ ', phi='+str(phi))
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#        'first_year_report/densityevolution,F=30,w=7p83,ph=0.pdf', 
#        format='pdf', bbox_inches='tight')
plt.show()
    
#%%
  
'''
Plot phases - doesnt look good
'''
import matplotlib.colors as col
norm = col.Normalize(vmin=-pi, vmax=pi) 

N = 51; A_site_start = 25;
a = 30; b=1; c=1; omega=7.83; phi=pi/2; T=2*pi/omega
centre = 25;
tspan = (0,10)
Nt = 100
t_eval = np.linspace(tspan[0], tspan[1], Nt)

psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;

sol = solve_ivp(lambda t,psi: F_OSC(t, psi, N, centre,
                                                     a,
                                                     omega, phi), 
                                    tspan, psi0, rtol=1e-7, atol=1e-7,
                                    t_eval=t_eval, method='RK45')



print('phi = '+str(phi))
print('omega = '+str(omega))

sz = 6
fig, ax = plt.subplots(figsize=(sz,sz))
ax.matshow(np.angle(sol.y), interpolation='none', cmap='PuOr', norm=norm)
#ax[3].matshow(np.angle(HF), interpolation='none', cmap='PuOr', norm=norm_phase)

x_positions = np.arange(0, Nt, T*(Nt/tspan[1]))
x_labels = list(range(len(x_positions)))
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
  labeltop=False)  
ax.set_xticks(x_positions)
ax.set_xlabel('T')
ax.set_xticklabels(x_labels)
ax.set_ylabel('site')

#fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm))
#fig.suptitle('psi evolution, oscillating single site potential a='
#             +str(a)+', omega='+str(omega))

plt.show()
    


#%%
'''
Put in report
'''
import numpy as np

N = 51; A_site_start = 25;
a = 30; 
b=0.1; c=0.1;
omega=5; T=2*pi/omega
centre = 25;
tspan = (0,10)
Nt = 100
sz = 14
fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))

titles = ['(a)\n', '(b)\n', '(c)\n', '(d)\n', '(e)\n', '(f)\n', ]
for nn, phi in enumerate([0, pi/2]):
    t_eval = np.linspace(tspan[0], tspan[1], Nt)
    
    psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
    
    sol = solve_ivp(lambda t,psi: F_MG(t, psi, N, centre,
                                                         a,
                                                         b,
                                                         c,
                                                         omega, phi), 
                                        tspan, psi0, rtol=1e-7, atol=1e-7,
                                        t_eval=t_eval, method='RK45')
    
    #sz = 7
    sz = 12

    ax[nn,0].matshow(abs(sol.y)**2, interpolation='none', cmap='Purples')
    ax[nn,1].matshow(np.real(sol.y), interpolation='none', cmap='Purples')
    ax[nn,2].matshow(np.angle(sol.y), interpolation='none', cmap='Purples')
    #ax[3].matshow(np.angle(HF), interpolation='none', cmap='PuOr', norm=norm_phase)
    ax[nn,0].set_title(titles[nn*3]+r'$|\psi(t)|^2$')
    ax[nn,1].set_title(titles[nn*3+1]+r'$\mathrm{Re}\{\psi(t)\}$')
    ax[nn,2].set_title(titles[nn*3+2]+r'$\mathrm{Imag}\{\psi(t)\}$')
    #ax[3].set_title('phase')
    x_positions = np.arange(0, Nt, T*(Nt/tspan[1]))
    x_labels = list(range(len(x_positions)))
    for i in range(3):
        ax[nn,i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[nn, i].set_xticks(x_positions)
        ax[nn,i].set_xlabel('T')
        ax[nn,i].set_xticklabels(x_labels)
        if i == 0:
            ax[nn,i].set_ylabel('site')
    
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.01, 0.7])
fig.colorbar(plt.cm.ScalarMappable(cmap='Purples'), cax=cbar_ax)# shrink=.5, pad=.01, aspect=10)
#fig.suptitle('F = '+str(a)+', omega='+str(omega)+ ', phi='+str(phi))
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#        'first_year_report/densityevolution,F=30,w=7p83.pdf', 
#        format='pdf', bbox_inches='tight')
plt.show()
    


#%%
import matplotlib
import seaborn as sns

size=10
params = {
            'legend.fontsize': 'small',
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
matplotlib.rcParams.update(params)

