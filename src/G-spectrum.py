# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:01:46 2021

@author: Georgia
"""
from numpy import  pi
import sys
sys.path.append('/Users/Georgia/Code/MBQD/floquet-simulations/src')
from hamiltonians import create_HF, getevalsandevecs, plotevecs
from hamiltonians import formatcomplex, roundcomplex
from hamiltonians import OrderEvecs, AlignEvecs
from scipy.special import jn_zeros
from scipy.linalg import eig as eig
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def phistring(phi):
    if phi == 0:
        return "0"
    elif phi == "phi":
        return r'$\phi$' 
    else:
        return  r'$\pi /$' + str(int(1/(phi/pi)))


size = 25
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

"""
Full Simulation
"""

N=49; centre=24; a=35; phi=pi/6;
omega=9.6

form='SS-p'
rtol=1e-11
UT, HF = create_HF(form, rtol, N, centre, a,phi, omega)
evals, evecs = getevalsandevecs(HF)
evecs = OrderEvecs(evecs, N)

func = np.abs
colour = "dodgerblue"
title = (func.__name__+"(evecs) ordered by " + "evals\nN="
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
for loop to compare eigenstates of different initial conditions
"""

N=49; centre=24; a=35; 
omega=9.6
form='SS-p'
rtol=1e-11
phis = [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
# phi=pi/7;
orderfunc = np.real
funcs = [np.abs, np.real, np.imag]

for phi in phis:
    UT, HF = create_HF(form, rtol, N, centre, a,phi, omega)
    evals, evecs = eig(HF)
    idx = orderfunc(evals).argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]

    fig, ax = plt.subplots(ncols=3, figsize=(5*1.5*3, 5))
    evec1 = roundcomplex(evecs[:,16],7)
    for nn in range(3):
        ax[nn].plot(range(N), funcs[nn](evec1), color=colour) 
    
    # fig.suptitle(""
    #                # +"Using theoretical G (hopping modified only)\n"
    #              +func.__name__+"(evecs) ordered by " + orderfunc.__name__ + "(evals)\nN="
    #              +str(N)+r", $V_{(n,n)} = $"
    #              + str(a)
    #              + r"$ \cos( $"
    #              + "{:.2f}".format(omega)
    #              + r'$ t$'
    #              + phistring(phi)
    #              + r'$) $'
    #              + ', rtol = '+str(rtol), y=0.934)#y=0.99)
    plt.show()



#%%

boundstate = 24
sz = 12
fig, ax = plt.subplots(nrows=3,figsize=(sz*1.6,sz), sharex=True)
funcs = [np.abs, np.real, np.imag]
for i in range(3):
    ax[i].plot(range(N), funcs[i](evecs[:,boundstate]), 'x', ms = 12, color="darkblue")  
    ax[i].set_title(funcs[i].__name__)

fig.suptitle("Bound State ("+r"$\epsilon = $"+ formatcomplex(evals[boundstate],9) + ")", y=0.96)
plt.show()



#%%
A_site_start = 36
psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
probi = np.square(np.abs([np.inner(np.conj(evecs[:,i]), psi0) for i in range(N)]))
IPR = np.sum(np.square(probi))
print(IPR)
fig, ax = plt.subplots(figsize=(6*2,6))
ax.plot(range(N), probi)
plt.show()
print(probi[boundstate])


#%%
#specific graph


N=49; centre=24; a=35; 
phis=[0, pi/3, pi/4, pi/7, pi/2];
ppp = a/jn_zeros(0,2)


omegas = np.linspace(30, 50, 7, endpoint=True)
omega = 9.6

# omega=9.6
form='SS-p'
rtol=1e-11
atomstarts = [ 28, 30, 35, 40]
orderfunc = np.real

sz = 4
fig, ax = plt.subplots(nrows=len(phis), ncols = len(atomstarts), 
                       figsize = (sz*len(atomstarts), sz*len(phis)))

for n1, phi in enumerate(phis):
    print(phi)
    UT, HF = create_HF(form, rtol, N, centre, a,phi, omega)
    evals, evecs = eig(HF)
    idx = orderfunc(evals).argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    for n2, astart in enumerate(atomstarts):
        psi0 = np.zeros(N, dtype=np.complex_); 
        psi0[astart] = 1;
        probi = np.square(np.abs([np.inner(np.conj(evecs[:,i]), psi0) for i in range(N)]))
        IPR = np.sum(np.square(probi))
        ax[n1, n2].plot(range(N), probi, color= 'darkmagenta')
        # ax[n1,n2].set_title("{:.2f}".format(IPR))
        if n2==0:
            ax[n1,n2].set_ylabel(r"$\phi=$"+phistring(phi), labelpad=50, rotation="horizontal")
        if n1==0:
            ax[n1,n2].set_title(r"$|\psi> = |$"+str(astart)+r">")

fig.text(0.5, 0.0001, r'$p_i$ for evecs {|i>}', ha='center')
fig.suptitle("N=" +str(N)+r", $V_{(n,n)} = $"
             + str(a)
             + r"$ \cos( $"
             + "{:.2f}".format(omega)
             + r'$ t$'
             + phistring("phi")
             + r'$) $'
             + ', rtol = '+str(rtol))
plt.tight_layout()
plt.show()

#%%

labels = [r'$p_i = |a_i|^2$', 
          r'$\mathrm{Real} \{  a_i\}$',
          r'$\mathrm{Imag} \{ a_i \}$']

N=49; centre=24; a=35; 
phis=[0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
ppp = a/jn_zeros(0,2)


omega = 9.6

# omega=9.6
form='SS-p'
rtol=1e-11
astart = 48
orderfunc = np.real
def squareabs(i):
    return np.square(np.abs(i))
funcs = [squareabs, np.real, np.imag]

evec_dist = np.zeros((len(phis),N))
sz = 6
fig, ax = plt.subplots(ncols=len(phis), nrows=len(funcs),
                       figsize = (sz*len(phis), sz*1.8))

for n1, phi in enumerate(phis):
    print(phi)
    UT, HF = create_HF(form, rtol, N, centre, a,phi, omega)
    evals, evecs = eig(HF)
    idx = orderfunc(evals).argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]

    psi0 = np.zeros(N, dtype=np.complex_); 
    psi0[astart] = 1;
    ai =[np.inner(np.conj(evecs[:,i]), psi0) for i in range(N)]
    probi = np.square(np.abs([np.inner(np.conj(evecs[:,i]), psi0) for i in range(N)]))
    evec_dist[n1] = probi
    
    for n2, func in enumerate(funcs):
        ax[n2, n1].plot(range(N), np.round(func(ai), 7))
        # ax[n1].set_title()
        if n2==0:
            ax[n2, n1].set_title(r"$\phi=$"+phistring(phi))
        if n1==0:
            ax[n2,n1].set_ylabel(labels[n2])
            
    
    

fig.text(0.5, 0.0001, r'evecs {|i>}', ha='center')
fig.suptitle("N=" +str(N)+r", $V_{(n,n)} = $"
             + str(a)
             + r"$ \cos( $"
             + "{:.2f}".format(omega)
             + r'$ t$'
             + phistring("phi")
             + r'$) $'
             + ', rtol = '+str(rtol) 
             + r", $|\psi(0)> = |$"+str(astart)+r">"
             +r"")
plt.tight_layout()
plt.show()

evec_dist = np.round(evec_dist, 7)
for i in range(len(phis)-1):
    print(np.all(evec_dist[i]==evec_dist[i+1]))


#%%


N=49; centre=24; a=35; 
phis=[0, pi/3, pi/4, pi/7, pi/2];
ppp = a/jn_zeros(0,2)

omegas = np.linspace(30, 50, 7, endpoint=True)

# omega=9.6
form='theoretical'
rtol=1e-11
atomstarts = [24, 25, 26, 27, 28, 30, 35, 40]
orderfunc = np.real

for omega in omegas:
    sz = 4
    fig, ax = plt.subplots(nrows=len(phis), ncols = len(atomstarts), 
                           figsize = (sz*len(atomstarts), sz*len(phis)))
    
    for n1, phi in enumerate(phis):
        print(phi)
        UT, HF = create_HF(form, rtol, N, centre, a,phi, omega)
        evals, evecs = eig(HF)
        idx = orderfunc(evals).argsort()[::-1]   
        evals = evals[idx]
        evecs = evecs[:,idx]
        
        for n2, astart in enumerate(atomstarts):
            psi0 = np.zeros(N, dtype=np.complex_); 
            psi0[astart] = 1;
            probi = np.square(np.abs([np.inner(np.conj(evecs[:,i]), psi0) for i in range(N)]))
            IPR = np.sum(np.square(probi))
            ax[n1, n2].plot(range(N), probi, color= 'darkmagenta')
            # ax[n1,n2].set_title("{:.2f}".format(IPR))
            if n2==0:
                ax[n1,n2].set_ylabel(r"$\phi=$"+phistring(phi), labelpad=50, rotation="horizontal")
            if n1==0:
                ax[n1,n2].set_title(r"$|\psi> = |$"+str(astart)+r">")
    
    fig.text(0.5, 0.0001, r'$p_i$ for evecs {|i>}', ha='center')
    fig.suptitle("N=" +str(N)+r", $V_{(n,n)} = $"
                 + str(a)
                 + r"$ \cos( $"
                 + "{:.2f}".format(omega)
                 + r'$ t$'
                 + phistring("phi")
                 + r'$) $'
                 + ', rtol = '+str(rtol))
    plt.tight_layout()
    plt.show()







