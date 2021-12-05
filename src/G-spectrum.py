# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:01:46 2021

@author: Georgia
"""

place = "Georgia"
from numpy import  pi
import sys
# C:\Users\Georgia Nixon\Code\MBQD\floquet-simulations\src
# sys.path.append("/Users/" + place + "/Code/MBQD/floquet-simulations/src")
sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
from hamiltonians import CreateHF,CreateHFGeneral, GetEvalsAndEvecsGen, plotevecs
from hamiltonians import formatcomplex, RoundComplex, PhiString
from hamiltonians import H0_PhasesNNHop
from hamiltonians import OrderEvecs, AlignEvecs
from hamiltonians import Cosine
from scipy.special import jn_zeros
from scipy.linalg import eig as eig
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


size = 12
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
Plot Evals and Evecs
"""

N=49; centre=24; a=35; rtol=1e-11
# centres = [6, 7, 25, 27, 29, 35]
# phases = [0, 1, 2, 3, 4, 5]
# phases = [0, 0, 0, 0, 0, 0]

phi1=0;
phiOffset = pi/2
phi2 = phi1+phiOffset
omega1=10#a /jn_zeros(0,1)[0]
omegaMultiplier = 2
omega2 = omega1*omegaMultiplier
T = 2*pi/omega1
circleBoundary = 0
onsite1 =0; onsite2=0

centres = [24, 25]
funcs = [Cosine, Cosine]
paramss = [[a, omega1, phi1, onsite1], [a, omega2, phi2, onsite2]]


# form = "SS-p"; hamiltonianString="$H(t)=H_0 + a \> \hat{n}_b \cos (\omega t + \phi) $"; paramsString = r"$a="+str(a)+r", \omega = "+"{:.2f}".format(omega)+", \phi = "+PhiString(phi)+r"$"
# form = "DS-p"; hamiltonianString = "$H(t)=H_0 + a \> \hat{n}_b \cos (\omega_1 t + \phi_1)  + a \> \hat{n}_{b+1} \cos (\omega_2 t + \phi_2)]$"; paramsString = r"$a=$"+str(a)+", "+r"$\omega_1="+"{:.2f}".format(omega1)+", \omega_2 = "+str(omegaMultiplier)+" \omega_1, \phi_1 = "+PhiString(phi1)+", \phi_2 = \phi_1 + \pi/2, N = "+str(N)+", b = "+str(centre)+"$ "
# form = "SSDF-p"; hamiltonianString = "$H(t)=H_0 + a \> \hat{n}_b [\cos (\omega_1 t + \phi_1)  +  \cos (\omega_2 t + \phi_2)]$"; paramsString = r"$a=$"+str(a)+", "+r"$\omega_1="+ "{:.2f}".format(omega1)+", \omega_2 = "+str(omegaMultiplier)+" \omega_1, \phi_1 ="+PhiString(phi1)+", \phi_2 = \phi_1 + \pi/2, N = "+str(N)+", b = "+str(centre)+"$ "
# form = "H0_PhasesNNHop"; hamiltonianString = "$H_0$"; paramsString=""


UT, HF = CreateHFGeneral(N, centres, funcs, paramss, T, circleBoundary)

# HF = H0_PhasesNNHop(N, centres, phases )
evals, evecs = GetEvalsAndEvecsGen(HF)
# evecs = OrderEvecs(evecs, N)

func = np.real
colour = "dodgerblue"

# title = ("real(evecs); "
#              + form +r";  "+hamiltonianString+"\n"
#              +paramsString)
title = ""


plotevecs(evecs, N, func, colour, title, ypos=0.955)    

sz=5
fig, ax = plt.subplots(figsize=(sz*1.4,sz))
ax.plot(range(N), func(evals), 'x', color=colour)
# fig.suptitle(r"evals;  "+ form +r";  "+hamiltonianString+"\n"
#              +paramsString, y=1)
plt.show()




#%%
from hamiltonians import Cosine, H0_Triangle

"""For different N's"""
N = 3; rtol=1e-11
centres= [1]
a = 35
omega = 8.1
T = 2*pi/omega
phi = 0
onsite = 0
funcs = [Cosine]
paramss = [[a, omega, phi, onsite]]
circleBoundary = 1


# J1 = 1 # once
# J2 = -1#exp(1j*pi/7) # twice
# H = H0_Triangle(J1, J2)
UT, HF = CreateHFGeneral(N, centres, funcs, paramss, T, circleBoundary)
HF2 = np.abs(HF)
# HF = np.abs(HF)
evals, evecs = GetEvalsAndEvecsGen(HF)
evals2, evecs2 = GetEvalsAndEvecsGen(HF2)

func = np.real
colour = "dodgerblue"


evalmin = np.min(np.array([np.min(evals), np.min(evals2)]))
evalmax = np.max(np.array([np.max(evals), np.max(evals2)]))

sz=4
fig, ax = plt.subplots(figsize=(sz*1.4,sz))
ax.plot(range(N), func(evals), 'x', color=colour)
# fig.suptitle(r"evals;  "+ form +r";  "+hamiltonianString+"\n"
#              +paramsString, y=1)
ax.set_ylim([evalmin-0.3, evalmax+0.3])
plt.show()



sz=4
fig, ax = plt.subplots(figsize=(sz*1.4,sz))
ax.plot(range(N), func(evals2), 'x', color=colour)
# fig.suptitle(r"evals;  "+ form +r";  "+hamiltonianString+"\n"
#              +paramsString, y=1)
ax.set_ylim([evalmin-0.3, evalmax+0.3])

plt.show()


sz = 3
fig, ax = plt.subplots(nrows = 1, ncols = N, sharex=True,
                        sharey=True,
                        figsize=(sz*N,sz*1))

for i in range(N):
    evec1 = evecs[:,i]
    ax[i].plot(range(N), func(evec1), color=colour, linewidth=0.6) 
ax[0].set_ylim([-1,1])
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
onsite = 0

sz = 4
fig, ax = plt.subplots(nrows=len(phis), ncols = len(atomstarts), 
                       figsize = (sz*len(atomstarts), sz*len(phis)))

for n1, phi in enumerate(phis):
    print(phi)
    UT, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)
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
            ax[n1,n2].set_ylabel(r"$\phi=$"+PhiString(phi), labelpad=50, rotation="horizontal")
        if n1==0:
            ax[n1,n2].set_title(r"$|\psi> = |$"+str(astart)+r">")

fig.text(0.5, 0.0001, r'$p_i$ for evecs {|i>}', ha='center')
fig.suptitle("N=" +str(N)+r", $V_{(n,n)} = $"
             + str(a)
             + r"$ \cos( $"
             + "{:.2f}".format(omega)
             + r'$ t$'
             + PhiString("phi")
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
    UT, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)
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
            ax[n2, n1].set_title(r"$\phi=$"+PhiString(phi))
        if n1==0:
            ax[n2,n1].set_ylabel(labels[n2])
            
    
    

fig.text(0.5, 0.0001, r'evecs {|i>}', ha='center')
fig.suptitle("N=" +str(N)+r", $V_{(n,n)} = $"
             + str(a)
             + r"$ \cos( $"
             + "{:.2f}".format(omega)
             + r'$ t$'
             + PhiString("phi")
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
        UT, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)
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
                ax[n1,n2].set_ylabel(r"$\phi=$"+PhiString(phi), labelpad=50, rotation="horizontal")
            if n1==0:
                ax[n1,n2].set_title(r"$|\psi> = |$"+str(astart)+r">")
    
    fig.text(0.5, 0.0001, r'$p_i$ for evecs {|i>}', ha='center')
    fig.suptitle("N=" +str(N)+r", $V_{(n,n)} = $"
                 + str(a)
                 + r"$ \cos( $"
                 + "{:.2f}".format(omega)
                 + r'$ t$'
                 + PhiString("phi")
                 + r'$) $'
                 + ', rtol = '+str(rtol))
    plt.tight_layout()
    plt.show()







