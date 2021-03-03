# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:01:46 2021

@author: Georgia
"""
from numpy import  pi
import sys
sys.path.append('/Users/Georgia/Code/MBQD/floquet-simulations/src')
from hamiltonians import create_HF
from scipy.special import jn_zeros
from scipy.linalg import eig as eig
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def phistring(phi):
    if phi == 0:
        return ""
    elif phi == "phi":
        return r'+ $\phi$' 
    else:
        return  r'$+ \pi /$' + str(int(1/(phi/pi)))
    
def roundcomplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j

def formatcomplex(num, dp):
    return ("{:."+str(dp)+"f}").format(num.real) + " + " + ("{:."+str(dp)+"f}").format(num.imag) + "i"



size = 23
params = {
            # 'legend.fontsize': size*0.75,
#          'figure.figsize': (20,8),
           'axes.labelsize': size,
           'axes.titlesize': size,
           "axes.edgecolor": "0.15",
            "axes.linewidth":1.25,
           'xtick.labelsize': size,
           'ytick.labelsize': size,
          'font.size': size,
          'xtick.bottom':True,
          'xtick.top':False,
          'ytick.left': True,
          'ytick.right':False,
          ## draw ticks on the left side
#          'axes.titlepad': 25
          # 'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
           'axes.grid':False,
          'font.family' : 'STIXGeneral',
          'mathtext.fontset':'stix',
          "axes.facecolor": '0.97',
           "axes.spines.left":   False,
            "axes.spines.bottom": False,
            "axes.spines.top":    False,
            "axes.spines.right":  False,
          }
mpl.rcParams.update(params)

#%%

N=49; centre=24; a=35; phi=pi/3;
omega=a/jn_zeros(0,1)[0]
# omega=9.6
form='SS-p'
rtol=1e-11
UT, HF = create_HF(form, rtol, N, centre, a,None, None,phi, omega)
evals, evecs = eig(HF)

func = np.imag
orderfunc = np.real

idx = orderfunc(evals).argsort()[::-1]   
evals = evals[idx]
evecs = evecs[:,idx]


sz = 2
num = 7
fig, ax = plt.subplots(nrows = num, ncols = num, sharex=True,
                       sharey=True,
                       figsize=(sz*num,sz*num))

for i in range(num):
    for j in range(num):
        evec1 = evecs[:,num*i + j]

        ax[i,j].plot( range(N), func(evec1), color='darkblue')   

fig.suptitle(func.__name__+"(evecs) ordered by " + orderfunc.__name__ + "(evals)\nN="
             +str(N)+r", $V_{(n,n)} = $"
             + str(a)
             + r"$ \cos( $"
             + "{:.2f}".format(omega)
             + r'$ t$'
             + phistring(phi)
             + r'$) $'
             + ', rtol = '+str(rtol), y=0.935)
plt.show()


fig, ax = plt.subplots(figsize=(6*1.4,6))
ax.plot(range(N), func(evals), 'x', color='darkblue')
fig.suptitle(func.__name__+"(evals)  ordered by " + orderfunc.__name__ + "(evals)")
plt.show()


#%%

sz = 12
fig, ax = plt.subplots(nrows=3,figsize=(sz*1.6,sz), sharex=True)
funcs = [np.abs, np.real, np.imag]
for i in range(3):
    ax[i].plot(range(N), funcs[i](evecs[:,48]), 'x', ms = 12, color="darkblue")  
    ax[i].set_title(funcs[i].__name__)

fig.suptitle("Bound State ("+r"$\epsilon = $"+ formatcomplex(evals[48],9) + ")", y=0.96)
plt.show()
