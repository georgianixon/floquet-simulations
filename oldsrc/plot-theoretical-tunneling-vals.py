# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:05:09 2021

@author: Georgia
"""


from scipy.special import jv, jn_zeros
from numpy import sin, exp, pi
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
          'mathtext.fontset':'stix',
           'font.family': 'STIXGeneral'
          })
import seaborn as sns
sns.set(style="darkgrid")
sns.set(rc={'axes.facecolor':'0.9'})
import numpy as np


def func(A, omega, phi):
    return exp(1j * A * sin(phi) / omega)* jv(0, A / omega)

def append(l1, l2, l3):
    return np.append(l1, np.append(l2, l3))

A = 35
phi = pi/7
omega = np.linspace(4, 20, 100)

res = func(A, omega, phi)

apply = [np.abs, np.real, np.imag]
labels = [r'$|H_{n,n+1}|$', 
          r'$\mathrm{Real}|H_{n,n+1}|$',
          r'$\mathrm{Imag}|H_{n,n+1}|$']

sz=30
fig, ax = plt.subplots(nrows=1, ncols=len(apply), figsize=(sz,sz/1.62/len(apply)),
                       constrained_layout=True, sharey=True)

turningvals = A/jn_zeros(0, 3)
integers = np.linspace(0, 5, 11)
coszeros = A*sin(phi)/(pi*(integers + 0.5))
sinzeros = A*sin(phi)/pi/integers[integers!=0]
ticks = np.array(list(map(lambda t: round(t, 2), append(turningvals,  coszeros, sinzeros))))
ticks = ticks[(ticks>4)&(ticks<20)]
 
for n1, f in enumerate(apply):
    ax[n1].plot(omega, f(res),                      
                    label=
                      r'$\phi=$'+str(round(phi/pi, 2))+r'$\pi$'
                    )

    ax[n1].set_xlabel(r'$\omega$', fontsize=20)
              
    ax[n1].set_title(labels[n1], fontsize=25)
    
    
    ax[n1].set_xticks(ticks)
    ax[n1].tick_params(axis='both', labelsize=16) 
    # ax[n1].grid(axis='y', color='0.2', linestyle='--', linewidth=2, alpha=0.2)
    # ax[n1].vlines(turningvals, -0.4, 0.4,  colors='0.5', linestyles='dotted')
    ax[n1].hlines(0, xmin=3.6, xmax=20.4, colors='0.5', linestyles='dotted' )
    ax[n1].set_xlim([3.6, 18])
    ax[n1].set_ylim([-0.43, 0.43])
                          
#cos zeros for real part
fig.suptitle( r'$V(t) = '+str(A)+r'\cos( \omega t + \pi /$' +
                 str(int(1/(phi/pi))) + 
                  r'$) $'+' |25><25|' , fontsize=30, fontfamily='STIXGeneral')


# ax[1].vlines(coszeros[(coszeros>4)&(coszeros<20)], -0.4, 0.4, colors='r', linestyles='dotted')
# ax[2].vlines(sinzeros[(sinzeros>4)&(sinzeros<20)], -0.4, 0.4, colors='r', linestyles='dotted')

plt.show()

#%%