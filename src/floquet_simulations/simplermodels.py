# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:20:31 2021

@author: Georgia
"""

import numpy as np
from scipy.linalg import eig
from numpy import sin, exp, pi
import matplotlib.pyplot as plt
from scipy.special import jv, jn_zeros
import matplotlib as mpl
import sys
sys.path.append('/Users/Georgia/Code/MBQD/floquet-simulations/src')
from hamiltonians import create_HF, hoppingHF
from hamiltonians import formatcomplex, roundcomplex
from hamiltonians import getevalsandevecs, plotevecs, plotevals
from hamiltonians import OrderEvecs, AlignEvecs

size = 25
params = {
            'legend.fontsize': size*0.75,
#          'figure.figsize': (20,8),
           'axes.labelsize': size,
           'axes.titlesize': size,
           "axes.edgecolor": "0.15",
            "axes.linewidth":1.25,
           'xtick.labelsize': size*0.75,
           'ytick.labelsize': size*0.75,
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
          "axes.facecolor": "0",#"1",#'0.97',
          "legend.facecolor":"0.9",
          "legend.fontsize":size,
           "axes.spines.left":   False,
            "axes.spines.bottom": False,
            "axes.spines.top":    False,
            "axes.spines.right":  False,
          }
mpl.rcParams.update(params)


CB91_Blue = 'darkblue'#'#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
red = "#FC4445"
dodgerblue = "dodgerblue"
slategrey = "slategrey"

def phistring(phi):
    if phi == 0:
        return "0"
    else:
        return  r'$\pi /$' + str(int(1/(phi/pi)))
    
def barrierHF(N, centre, a, h):
    HF = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    HF[centre][centre] = h
    return HF


funcs = [np.real, np.abs, np.imag]
colours = ["darkblue", "#13546A",'#b21e4f']
    
#%%
"""
Full Simulation
"""

N=49; centre=24; a=35; phi=pi/7;
omega=9.6

form='SS-p'
rtol=1e-11
UT, HF = create_HF(form, rtol, N, centre, a,phi, omega)
evals, evecs = getevalsandevecs(HF)

func = np.real
colour = "dodgerblue"
title = (func.__name__+"(evecs) ordered by " + "real(evals)\nN="
             +str(N)+r", $V_{(n,n)} = $"
             + str(a)
             + r"$ \cos( $"
             + "{:.2f}".format(omega)
             + r'$ t$'
             + " + " + phistring(phi)
             + r'$) $'
             + ', rtol = '+str(rtol))


plotevecs(evecs, N, func, colour, title, ypos=0.955)    
    

fig, ax = plt.subplots(figsize=(6*1.4,6))
ax.plot(range(N), func(evals), 'x', color=colour)
fig.suptitle(func.__name__+"(evals)  ordered by " + "real(evals)")
plt.show()

#%%

"""
Barrier
"""
N = 49
centre = 24
a=35; h = 1;

HF = barrierHF(N, centre, a, h)
evals, evecs = getevalsandevecs(HF)

for func, colour in zip(funcs, colours):
    title = (func.__name__+"(evecs) ordered by " 
             + "real(evals)\nN="
                     +str(N)
                       +r", $V = $"
                       +str(h)
                       +r"$|24><24|$")

    plotevecs(evecs, N, func, colour, title,0.934 )


title = ("evals"+ ", N="+str(N)+", barrier="+str(h))
plotevals(evals, N, title)

#%%

"""
Hopping Only
"""


N = 49
centre = 24
a=35; 
# omega=a/jn_zeros(0,1)[0];
omega=9.6;
phi = pi/7;

HF, entry = hoppingHF(N, centre, a, omega, phi)
evals, evecs = getevalsandevecs(HF)

for func, colour in zip(funcs, colours):
    title = (""
             +func.__name__+"(evecs) ordered by " 
             + "real(evals)\nN="
             +str(N)
              +r", tunneling "
              +r"$= \exp (i a \sin( \phi / \omega) * \mathcal{J}_0(a / \omega) )$"
              +r" = " + formatcomplex(entry, 3)
              + ", a = "+str(a)+ r", $\omega = $" +"{:.2f}".format(omega)
              +r", $\phi = $"+phistring(phi))
    plotevecs(evecs, N, func, colour, title)


title = ("evals, N="+str(N)+", a = "+str(a)
             + r", $\omega = $" +"{:.2f}".format(omega)
              +"\n"+r"T "
              +r"$= \exp (i a \sin( \phi / \omega) * \mathcal{J}_0(a / \omega) )$")
plotevals(evals, N, title)


#%%

"""
Numerical scenario
"""

N=49; centre=24; a=35; phi=pi/2;
omega=a/jn_zeros(0,1)[0]
# omega=9.6
form='SS-p'
rtol=1e-11
UT, HF = create_HF(form, rtol, N, centre, a,phi, omega)


evals, evecs = getevalsandevecs(HF)

for func, colour in zip(funcs, colours):
    title = (func.__name__+"(evecs) ordered by " + "real(evals)\nN="
             +str(N)
              + ", a = "+str(a)+ r", $\omega = $" +"{:.2f}".format(omega)
              +r", $\phi = $"+phistring(phi)
              +", full numerical simulation")
    plotevecs(evecs, N, func, colour, title)

title = ("evals, full numerical sim\nN="+str(N)+
             ", a = "+str(a)+ r", $\omega = $" +"{:.2f}".format(omega))

#%%

"""
Comparing simpler models
"""


N=49; centre=24; a=35; phi=0;
omega = 9.6
omega=a/jn_zeros(0,1)[0]-0.1
# omega=a/jn_zeros(0,1)[0]
# omega=9.6
form='SS-p'
rtol=1e-11

_, HFSS = create_HF(form, rtol, N, centre, a,phi, omega)
HFHO, entry = hoppingHF(N, centre, a, omega, phi)

evalsSS, evecsSS = getevalsandevecs(HFSS)
evalsHO, evecsHO = getevalsandevecs(HFHO)
evecsSS = OrderEvecs(evecsSS, N)
evecsHO = OrderEvecs(evecsHO, N)
# evecsHO = AlignEvecs(evecsSS, evecsHO, N)

# for i in range(N):
#     if np.all(np.real(evecsHO[:,i])==-np.real(evecsSS[:,i])):
#         evecsSS[:,i] = -evecsSS[:,i]
        
    
func = np.abs
colourSS = "dodgerblue"#"#613DC1"#
colourHO = "1"#"0.7"#'#9D2EC5'

sz = 4
num = 7
fig, ax = plt.subplots(nrows = num, ncols = num, sharex=True,
                       sharey=True,
                       figsize=(sz*num*1.6,sz*num))

for i in range(num):
    for j in range(num):
        evec1SS = evecsSS[:,num*i + j]
        evec1HO = evecsHO[:,num*i + j]

        ax[i,j].plot(range(N), func(evec1SS), color=colourSS,
                     # label="single site oscillation", 
                     label="H(t)",
                     linewidth = 5)
        ax[i,j].plot(range(N), func(evec1HO), color=colourHO, 
                      label="Toy model; tunnelling only")
                      # label="hopping toy model")

handles_legend, labels_legend = ax[0,0].get_legend_handles_labels()    
fig.legend(handles_legend, labels_legend, loc="right")
# fig.suptitle(func.__name__+"(evecs) ordered by " + "real(evals)\nN="
#              +str(N)
#               + ", a = "+str(a)+ r", $\omega = $" +"{:.2f}".format(omega)
#               +r", $\phi = $"+phistring(phi)
#               +", effective tunneling = "+formatcomplex(entry, 3)
#               , y=0.934)

fig.suptitle("Abs(evecs)\n"
              + "A = "+str(a)+ r", $\omega = $" +"{:.2f}".format(omega)
              +r", $\phi = $"+phistring(phi)
              , y=0.934)
plt.show()


assert(np.all(0 == np.imag(evalsSS)))
assert(np.all(0 == np.imag(evalsHO)))
sz = 8
fig, ax = plt.subplots(figsize=(sz*1.4,sz))
ax.plot(range(N), np.real(evalsSS), 'x', markersize=10, color=colourSS, 
        label="H(t)")
        # label="single site oscillation")
ax.plot(range(N), np.real(evalsHO),  'o', markersize = 3, color=colourHO, 
        label="Toy model; tunnelling only")
fig.legend()
# fig.suptitle(title, y=ypos)
plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import random
# Plot color palette
def plot_color_palette(ColList):

    figure = sns.palplot(ColList, size=2)
    plt.xlabel(";   ".join(ColList), fontsize=20) 
    plt.show(figure)

plot_color_palette([CB91_Blue, CB91_Green, CB91_Pink, CB91_Purple, CB91_Violet, CB91_Amber, red, dodgerblue, slategrey])


#%%

plot_color_palette(["#437093", "#C04C48", "#F5B14C", "#00A89D"])




