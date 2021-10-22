
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:49 2020
|
@author: Georgia
"""

place = "Georgia Nixon"
from numpy import exp, sin, cos, pi, log, sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import SolveSchrodinger, SolveSchrodingerGeneral, SolveSchrodingerTimeIndependent
from hamiltonians import  PhiString

from hamiltonians import H_PhasesOnLoopsOneD, H_0, H_0_Phases

import matplotlib as mpl
import seaborn as sns
from scipy.special import jv, jn_zeros
from fractions import Fraction

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/"

size=28
params = {
            'legend.fontsize': size*0.65,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.65,
          'ytick.labelsize': size*0.65,
          'font.size': size,
          'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
          'axes.grid':False,
          'font.family' : 'STIXGeneral',
          'mathtext.fontset':'stix',
          # "grid.alpha":1
          }


mpl.rcParams.update(params)


def PlotPsi(psi, x_positions, x_labels, title, normaliser):
    """
    Plot Matter Wave
    """
    
    mpl.rcParams.update({
          'mathtext.fontset':'stix'
          })
    
    apply = [lambda x: np.abs(x)**2, np.real, np.imag]
    labels = [r'$|\psi(t)|^2$', r'$\mathrm{Re}\{\psi(t)\}$', r'$\mathrm{Imag}\{\psi(t)\}$']
    
    
    cmapcol = 'PuOr' #PiYG_r
    cmap= mpl.cm.get_cmap(cmapcol)
    
    sz = 6
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                            figsize=(sz*len(apply)*1.3,sz))
    
    for i, f in enumerate(apply):
        ax[i].matshow(f(psi), interpolation='none', cmap=cmap, norm=normaliser, aspect='auto')
        ax[i].set_title(labels[i],  fontfamily='STIXGeneral')
        ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[i].set_xticks(x_positions)
        ax[i].set_xlabel('t/T', fontfamily='STIXGeneral')
        ax[i].set_xticklabels(x_labels)
        for side in ["bottom", "top", "left", "right"]:
            ax[i].spines[side].set_visible(False)
        if i == 0:
            ax[i].set_ylabel('site', fontfamily='STIXGeneral')
    
    cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmapcol, norm=normaliser), cax=cax)
    fig.suptitle(title, y = 1.2,  fontfamily='STIXGeneral')
    
    plt.show()
    
def PlotProbCurrent(psi, x_positions, x_labels, title, normaliser):
    """
    Plot Probability current
    needs real, abs
    """
    
    mpl.rcParams.update({
          'mathtext.fontset':'stix'
          })
    
    apply = [np.real]
    labels = [ r'$\mathrm{Re}\{j(x,t)\}$']
    
    
    cmapcol = 'PuOr' #PiYG_r
    cmap= mpl.cm.get_cmap(cmapcol)
    
    sz = 7
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                            figsize=(sz*len(apply),sz))
    

    ax.matshow(np.real(psi), interpolation='none', cmap=cmap, norm=normaliser, aspect='auto')
    ax.set_title(labels[0],  fontfamily='STIXGeneral')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xticks(x_positions)
    ax.set_xlabel('t/T', fontfamily='STIXGeneral')
    ax.set_xticklabels(x_labels)
    for side in ["bottom", "top", "left", "right"]:
        ax.spines[side].set_visible(False)

    ax.set_ylabel('site', fontfamily='STIXGeneral')
    
    cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmapcol, norm=normaliser), cax=cax)
    fig.suptitle(title, y = 1.1,  fontfamily='STIXGeneral')
    
    plt.show()
    
def PlotTwoPsi(psi1, psi2, x_positions, x_labels, title, normaliser):
    """
    Plot the difference between two wavefunctison
    |psi1|^2 - |psi2|^2
    Re[psi1 - psi2]
    Im[psi1 - psi2]

    """
    
    apply = [lambda x,y: np.abs(x)**2 - np.abs(y)**2, 
             lambda x,y: np.abs(x - y)**2, 
             lambda x,y: np.real(x - y),
             lambda x,y: np.imag(x - y)]
    labels = [r'$|\psi_1(t)|^2 - |\psi_2(t)|^2$', 
              r'$|\psi_1(t) - \psi_2(t)|^2$',
              r'$\mathrm{Re}\{\psi_1(t) - \psi_2(t)\}$', r'$\mathrm{Imag}\{\psi_1(t) - \psi_2(t)\}$']
    
    
    cmapcol = 'PuOr' #PiYG_r
    cmap= mpl.cm.get_cmap(cmapcol)
    
    sz = 5
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                            figsize=(sz*len(apply),sz))
    
    for i, f in enumerate(apply):
        ax[i].matshow(f(psi1, psi2), interpolation='none', cmap=cmap, norm=normaliser, aspect='auto')
        ax[i].set_title(labels[i])
        ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[i].set_xticks(x_positions)
        ax[i].set_xlabel('t/T')
        ax[i].set_xticklabels(x_labels)
        for side in ["bottom", "top", "left", "right"]:
            ax[i].spines[side].set_visible(False)
        if i == 0:
            ax[i].set_ylabel('site')
    
    cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmapcol, norm=normaliser), cax=cax)
    fig.suptitle(title, y = 1.2)
    plt.show()
    

    
# def PhiStringNum(phi):
#     if phi == 0:
#         return "0"
#     elif phi == "phi":
#         return r"$\phi$" 
#     else:
#         return  r"\pi /" + str(int(1/(phi/pi)))
    
def Differentiate(array, dx=1, circleBoundary = 0):
    xDiff = np.zeros(len(array), dtype=np.complex128)
    if circleBoundary:
        xDiff[0] = (array[1] - array[-1])/dx
        xDiff[-1] = (array[0] - array[-2])/dx
    else:
        xDiff[0] = (array[1] - array[0])/dx
        xDiff[-1] = (array[-1] - array[-2])/dx
    for i in range(len(array)-2):
        xDiff[i+1] = (array[i+2] - array[i])/2/dx
    return xDiff

def ProbCurrentofPsi(psi, circleBoundary=0):
    nTimes = np.shape(psi)[1]
    psiDiff =  np.array([Differentiate(psi[:,i], circleBoundary=circleBoundary) for i in range(nTimes)]).T
    psiConjDiff = np.array([Differentiate(np.conj(psi[:,i]), circleBoundary=circleBoundary) for i in range(nTimes)]).T
    probCurrent = (1/2/1j)*np.array([np.multiply(np.conj(psi[:,i]), psiDiff[:,i])
                                  - np.multiply(psi[:,i], psiConjDiff[:,i]) for i in range(nTimes)]).T
    assert(np.all(np.imag(probCurrent)==0))
    probCurrent = np.real(probCurrent)
    return probCurrent
    


#%%
"""
Compare two matter waves
"""


""""Global parameters"""
N = 92#182; 
centre = 45#90;
rtol=1e-11

#SS-p params
a = 35
omega = 4#a /jn_zeros(0,1)[0]
phi = 0
T = 2*pi / omega
form = "SS-p"; hamiltonianString="$H(t)=H_0 + a \> \hat{n}_b \cos (\omega t + \phi) $"; paramsString = r"$a="+str(a)+r", \omega = "+"{:.2f}".format(omega)+", \phi = "+PhiString(phi)+r"$"


form = "SS-p"; 
hamiltonianString="$H(t)=H_0 + a \> \hat{n}_b \cos (\omega t + \phi) $"; 
paramsString = r"$a="+str(a)+r", \omega = "+"{:.2f}".format(omega)+", \phi = "+PhiString(phi)+r"$"



form = "DS-p"; 
hamiltonianString = (r"$H(t)=H_0 + \hat{n}_b [a \> \cos (\omega_1 t + \phi_1) + s_1]  + "
                      +r"\hat{n}_{b+1} [a \> \cos (\omega_2 t + \phi_2) + s_2]$"); 
paramsString = (r"$a="+str(a)+", "+r"\omega_1="+"{:.2f}".format(omega1)+
                ", \omega_2 = "+str(omegaMultiplier)+" \omega_1, \phi_1 = "+PhiString(phi1)
                +", \phi_2 = \phi_1 + \pi/2, s_1 = " + str(onsite1)+r", s_2 = "+ str(onsite2)
                + r", N = "+str(N)+", b = "+str(centre)+"$ ")
form = "SSDF-p"; hamiltonianString = "$H(t)=H_0 + a \> \hat{n}_b [\cos (\omega_1 t + \phi_1)  +  \cos (\omega_2 t + \phi_2)]$"; paramsString = r"$a=$"+str(a)+", "+r"$\omega_1="+ "{:.2f}".format(omega1)+", \omega_2 = "+str(omegaMultiplier)+" \omega_1, \phi_1 ="+PhiString(phi1)+", \phi_2 = \phi_1 + \pi/2, N = "+str(N)+", b = "+str(centre)+"$ "


# #DS-p params
# form = "DS-p"; 
# phiOffset=pi/2
# phi2=phi1+phiOffset
# onsite1 = 0
# onsite2 = 20
# onsite = [onsite1, onsite2]
# omega1= 9.1#a/jn_zeros(0,1)[0]
# omegaMultiplier=2
# omega2=omega1*omegaMultiplier
# phi=[phi1,phi2]
# omega=[omega1,omega2]
# T=2*pi/omega1
# hamiltonianString = (r"$H(t)=H_0 + \hat{n}_b [a \> \cos (\omega_1 t + \phi_1) + s_1]  + "
#                       +r"\hat{n}_{b+1} [a \> \cos (\omega_2 t + \phi_2) + s_2]$"); 
# paramsString = (r"$a="+str(a)+", "+r"\omega_1="+"{:.2f}".format(omega1)+
#                 ", \omega_2 = "+str(omegaMultiplier)+" \omega_1, \phi_1 = "+PhiString(phi1)
#                 +", \phi_2 = \phi_1 + \pi/2, s_1 = " + str(onsite1)+r", s_2 = "+ str(onsite2)
#                 + r", N = "+str(N)+", b = "+str(centre)+"$ ")

# form = "SSDF-p"; 
# hamiltonianString = "$H(t)=H_0 + a \> \hat{n}_b [\cos (\omega_1 t + \phi_1)  +  \cos (\omega_2 t + \phi_2)]$"; 
# paramsString = r"$a=$"+str(a)+", "+r"$\omega_1="+ "{:.2f}".format(omega1)+", \omega_2 = "+str(omegaMultiplier)+" \omega_1, \phi_1 ="+PhiString(phi1)+", \phi_2 = \phi_1 + \pi/2, N = "+str(N)+", b = "+str(centre)+"$ "




#TS params
# a = 35
# phi1=0;
# phiOffset2=pi/2
# phiOffset3 = pi/4
# phi2=phi1+phiOffset2
# phi3 = phi1+phiOffset3
# phi=[phi1,phi2, phi3]
# onsite1 = 0
# onsite2 = 20
# onsite3 = 5
# onsite = [onsite1, onsite2, onsite3]
# omega1= a/jn_zeros(0,1)[0]
# omegaMultiplier2=2
# omegaMultiplier3 = 3
# omega2=omega1*omegaMultiplier2
# omega3 = omega1*omegaMultiplier3
# omega=[omega1,omega2, omega3]
# T=2*pi/min(omega)
# form = "TS-p"; 
# hamiltonianString = (r"$H(t)=H_0 + \hat{n}_b [a \> \cos (\omega_1 t + \phi_1) + s_1]  + "
#                       +r"\hat{n}_{b+1} [a \> \cos (\omega_2 t + \phi_2) + s_2] + "
#                       +r"\hat{n}_{b+2} [a \> \cos (\omega_3 t + \phi_3) + s_3]  $"); 
# paramsString = (r"$a="+str(a)+", "+r"\omega_1="+"{:.2f}".format(omega1)+
#                 ", \omega_2 = "+str(omegaMultiplier2)+" \omega_1, \omega_3 = "+str(omegaMultiplier3)+"\omega_1, "
#                 +r"\phi_1 = "+PhiString(phi1) +", \phi_2 = \phi_1 + "+PhiString(phiOffset2) + r", \phi_3 = \phi_1 + "+PhiString(phiOffset3)
#                 +r", s_1 = " + str(onsite1)+r", s_2 = "+ str(onsite2) +r", s_3 = "+str(onsite3)
#                 + r", N = "+str(N)+", b = "+str(centre)+"$ ")

"""wave 1 parameters""" # for ssdf
A_site_start1 = 40#85;
"""wave 2 params"""
A_site_start2 = 51#96;


# plot potential
t = np.linspace(0, 4*2*pi/omega1, 100)
shake1 = a*cos(omega1*t + phi1) + onsite1
shake2 = a*cos(omega2*t + phi2) + onsite2
shake3 = a*cos(omega3*t + phi3) + onsite3
plt.plot(t, shake1+shake2+shake3, label="V(t)")
plt.plot(t, np.abs(shake1+shake2+shake3), label = "abs(V(t))")
plt.ylabel("V")
plt.xlabel("t")
plt.legend()
plt.show()



"""solver params"""
nOscillations = 30
#how many steps we want. NB, this means we will solve for nTimesteps+1 times (edges)
nTimesteps = nOscillations*100
n_osc_divisions = 2
tspan = (0,nOscillations*T)
t_eval = np.linspace(tspan[0], tspan[1], nTimesteps)





psi0_1 = np.zeros(N, dtype=np.complex_); psi0_1[A_site_start1] = 1;
psi0_2 = np.zeros(N, dtype=np.complex_); psi0_2[A_site_start2] = 1;
psi1 = SolveSchrodinger(form, rtol, N, centre, a, omega, phi, 
                                  tspan, nTimesteps, psi0_1, onsite)
psi2 = SolveSchrodinger(form, rtol, N, centre, a,omega, phi,
                                  tspan, nTimesteps, psi0_2, onsite)



"""plot"""


# normaliser = mpl.colors.Normalize(vmin=-1, vmax=1)
linthresh = 1e-2
normaliser=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
x_positions = np.linspace(0, nTimesteps, int(nOscillations/n_osc_divisions+1))
x_labels = list(range(0, nOscillations+1, n_osc_divisions))



#flip one
psi2 = np.flip(psi2, axis=0)

PlotPsi(psi1, x_positions, x_labels,  form+", "+hamiltonianString+"\n"+paramsString+r"$,\> \psi_{start site} = " +str(A_site_start1)+r"$",
      normaliser)
PlotPsi(psi2, x_positions, x_labels,  form+", "+hamiltonianString+"\n"+paramsString+r"$,\> \psi_{start site} = " +str(A_site_start2)+r"$",
      normaliser)


#plot difference
title = form+", "+hamiltonianString+"\n"+paramsString+r"$,\>\psi_{start site 1}=" +str(A_site_start1)+r",\> \psi_{start site 2} = " +str(A_site_start2)+r"$"
PlotTwoPsi(psi1, psi2, x_positions, x_labels, title,
      normaliser)


""" Is there a diode? """


shakeLen = len(omega)

psi1AboveBorder = psi1[:centre,:-1]
psi2AboveBorder = psi2[:centre,:-1]

psi1BelowBorder = psi1[centre+shakeLen:,:-1]
psi2BelowBorder = psi2[centre+shakeLen:,:-1]
psi1AtShake = psi1[centre:centre+shakeLen,:-1]
psi2AtShake = psi2[centre:centre+shakeLen,:-1]


psiDiffAboveBorder = np.abs(psi1AboveBorder)**2 - np.abs(psi2AboveBorder)**2
psiDiffBelowBorder = np.abs(psi1BelowBorder)**2 - np.abs(psi2BelowBorder)**2
psiDiffAtShake = np.abs(psi1AtShake)**2 - np.abs(psi2AtShake)**2
psiDiff =  np.abs(psi1[:,:-1])**2 - np.abs(psi2[:,:-1])**2
psiOverallDiffAboveBorder = np.sum(psiDiffAboveBorder, axis=0)
psiOverallDiffBelowBorder = np.sum(psiDiffBelowBorder, axis=0)
psiOverallDiff = np.sum(psiDiff, axis=0)
psiDiffAtShake = np.sum(psiDiffAtShake, axis=0)

fig, ax = plt.subplots(figsize = (12,8))
plt.plot(t_eval/T, psiOverallDiffAboveBorder)
plt.title("Psi Diff Above Border [:45]\n"+form+", "+paramsString, y=1.02)
plt.show()


fig, ax = plt.subplots(figsize = (12,8))
plt.plot(t_eval/T, psiDiffAtShake)
plt.title("Psi Diff At Shake [45:48]\n"+form+", "+paramsString, y=1.02)
plt.show()


fig, ax = plt.subplots(figsize = (12,8))
plt.plot(t_eval/T, psiOverallDiffBelowBorder)
plt.title("Psi Diff Below Border [48:]\n"+form+", "+paramsString, y=1.06)
plt.show()

# fig, ax = plt.subplots(figsize = (12,8))
# plt.plot(t_eval, psiOverallDiff)
# plt.title("Psi Overall Diff\n"+form+", "+paramsString)
# plt.show()


    
#%%


"""
Homogeneous expansion
"""


# revolving phases
# form = "H0T"
# N = 91; 
# centre = 45;
# rtol=1e-11
# T = 1
# psi2 = SolveSchrodinger("H0T", N, centre, rtol, tspan, nTimesteps, psi0)


# phases with non trivial loops and NNN hopping

N = 211
T = 1
centre = 105
aSiteStart1 = 100; aSiteStart2 = 110
psi01 = np.zeros(N, dtype=np.complex_); psi01[aSiteStart1] = 1;
psi02 = np.zeros(N, dtype=np.complex_); psi02[aSiteStart2] = 1;
p0 = pi/7; p1 = pi/3; p2 = pi/2; p3 = pi/4
# HPhases = H_PhasesOnLoopsOneD(N, centre, 0, pi/3, pi/4, np.nan,np.nan)
HPhases = H_0_Phases(N, [99, 107, 120, 140], [pi/4, pi/3, pi/7, pi/sqrt(3)])
H0 = H_0(N)

# HPhasesFlip = np.flip(np.flip(HPhases, 0).T, 0)

nOscillations = 50
#how many steps we want. NB, this means we will solve for nTimesteps+1 times (edges)
nTimesteps = nOscillations*100
n_osc_divisions = 2
tspan = (0,nOscillations*T)
t_eval = np.linspace(tspan[0], tspan[1], nTimesteps)


psi1 = SolveSchrodingerTimeIndependent(HPhases, tspan, nTimesteps, psi01)
psi2 = SolveSchrodingerTimeIndependent(H0, tspan, nTimesteps, psi01)

pMax = np.max(np.abs(np.vstack((psi1, psi2))))
pMin = np.min(np.vstack((np.real(np.vstack((psi1, psi2))), np.imag(np.vstack((psi1, psi2))))))
absMax = np.max([np.abs(pMax), np.abs(pMin)])
        
normaliser = mpl.colors.Normalize(vmin=-absMax, vmax=absMax)
linthresh = 1e-3
normaliser=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-absMax, vmax=absMax, base=10)

x_positions = np.linspace(0, nTimesteps, int(nOscillations/n_osc_divisions+1))
x_labels = list(range(0, nOscillations+1, n_osc_divisions))

PlotPsi(psi1, x_positions, x_labels, r"title",
      normaliser)

#flip one
# psi2 = np.flip(psi2, axis=0)

PlotPsi(psi2, x_positions, x_labels,   r"title",
      normaliser)





#plot difference
PlotTwoPsi(psi1, psi2, x_positions, x_labels, r"title", normaliser)

probCurrent1 = ProbCurrentofPsi(psi1, circleBoundary=0)
probCurrent2 = ProbCurrentofPsi(psi2, circleBoundary=0)


PlotProbCurrent(probCurrent1, x_positions, x_labels,  r"prob current $\psi_1$",
      normaliser)
PlotProbCurrent(probCurrent2, x_positions, x_labels,  r"prob current $\psi_2$",
      normaliser)

#difference between probability currents
# PlotProbCurrent(probCurrent1 - probCurrent2, x_positions, x_labels, "prob current difference", normaliser)


""" Look at probability current overall """

totalCurrent1 = np.sum(probCurrent1, axis=0)[:-1]
totalCurrent2 = np.sum(probCurrent2, axis=0)[:-1]


fig, ax = plt.subplots(figsize = (16,8))
plt.plot(t_eval/T, totalCurrent1, label=r"$\sum_x j(x,t)_1$")
plt.plot(t_eval/T, totalCurrent2, label=r"$\sum_x j(x,t)_2$")
plt.xlabel("t/T")
plt.legend()
plt.title("total probability current", y=1.02)
plt.show()




""" Is there a diode? """

psi1AboveBorder = psi1[:centre-2,:-1]
psi2AboveBorder = psi2[:centre-2,:-1]

psi1BelowBorder = psi1[centre+3:,:-1]
psi2BelowBorder = psi2[centre+3:,:-1]
psi1AtShake = psi1[centre-2:centre+3,:-1]
psi2AtShake = psi2[centre-2:centre+3,:-1]


psiDiffAboveBorder = np.abs(psi1AboveBorder)**2 - np.abs(psi2AboveBorder)**2
psiDiffBelowBorder = np.abs(psi1BelowBorder)**2 - np.abs(psi2BelowBorder)**2
psiDiffAtShake = np.abs(psi1AtShake)**2 - np.abs(psi2AtShake)**2
psiDiff =  np.abs(psi1[:,:-1])**2 - np.abs(psi2[:,:-1])**2
psiOverallDiffAboveBorder = np.sum(psiDiffAboveBorder, axis=0)
psiOverallDiffBelowBorder = np.sum(psiDiffBelowBorder, axis=0)
psiOverallDiff = np.sum(psiDiff, axis=0)
psiDiffAtShake = np.sum(psiDiffAtShake, axis=0)

fig, ax = plt.subplots(figsize = (12,8))
plt.plot(t_eval/T, psiOverallDiffAboveBorder)
plt.title("Psi Diff Above Border [:"+str(centre)+"]", y=1.02)
plt.show()


fig, ax = plt.subplots(figsize = (12,8))
plt.plot(t_eval/T, psiDiffAtShake)
plt.title("Psi Diff At Shake ["+str(centre)+"]", y=1.02)
plt.show()


fig, ax = plt.subplots(figsize = (12,8))
plt.plot(t_eval/T, psiOverallDiffBelowBorder)
plt.title("Psi Diff Below Border ["+str(centre)+":]", y=1.06)
plt.show()

# fig, ax = plt.subplots(figsize = (12,8))
# plt.plot(t_eval, psiOverallDiff)
# plt.title("Psi Overall Diff\n"+form+", "+paramsString)
# plt.show()


#%%
""" Ramp """

"""General Params"""  
N = 92#182; 
rtol=1e-11

"""wave 1 parameters""" # for ssdf
A_site_start1 = 40#85;
"""wave 2 params"""
A_site_start2 = 51#96;

"""Ramp params"""
def Ramp(params, t): # ramp
    a = params[0]
    omega = params[1]
    phi = params[2]
    onsite = params[3]
    
    nCycle = np.floor(t*omega/2/pi + phi/2/pi)
    y = a*omega*t/2/pi + a*phi/2/pi - nCycle*a + onsite
    return y 

def RampHalf(params, t): # ramp
    a = params[0]
    omega = params[1]
    phi = params[2]
    onsite = params[3]
    
    nHalfCycle = np.floor(t*omega/pi + phi/pi)
    y = (a*omega*t/pi + a*phi/pi - nHalfCycle*a)*((nHalfCycle + 1 ) % 2) + onsite
    return y 


"""Blip params"""
def Blip(params, t):
    a = params[0]
    omega = params[1]
    phi = params[2]
    onsite = params[3]
    
    nHalfCycle = np.floor(t*omega/pi + phi/pi)
    y = a*sin(omega*t + phi)*((nHalfCycle+1) % 2) + onsite
    return y

"""Usual Cos shake"""
def Cosine(params, t):
    a = params[0]
    omega = params[1]
    phi = params[2]
    onsite = params[3]
    y = a*cos(omega*t + phi)+ onsite
    return y 


a = 35
omega1=10; omega2=2*omega1; omega3=omega1
phi1=0; phi2=pi/4; phi3=pi/2
onsite1=0; onsite2=5; onsite3=10
T=2*pi/omega1
centres = [45,46, 47]
funcs = [RampHalf, RampHalf, RampHalf]

params = [[a, omega1, phi1, onsite1], [a, omega2, phi2, onsite2], [a, omega3, phi3, onsite3]]



t = np.linspace(0, 4*2*pi/omega1, 100)
shake1 = funcs[0](params[0], t)
shake2 = funcs[1](params[1], t)
shake3 = funcs[2](params[2], t)
plt.plot(t, shake1+shake2+shake3)
plt.plot(t, np.abs(shake1+shake2+shake3))
plt.ylabel("V")
plt.xlabel("t")
plt.show()

plt.plot(t, shake1, label="site 1")
plt.plot(t, shake2, label="site 2")
plt.plot(t, shake3, label="site 3")
plt.ylabel("V")
plt.xlabel("t")
plt.legend()
plt.show()



"""solver params"""
nOscillations = 30
#how many steps we want. NB, this means we will solve for nTimesteps+1 times (edges)
nTimesteps = nOscillations*100
n_osc_divisions = 2
tspan = (0,nOscillations*T)
t_eval = np.linspace(tspan[0], tspan[1], nTimesteps)


"""Solve"""
psi0_1 = np.zeros(N, dtype=np.complex_); psi0_1[A_site_start1] = 1;
psi0_2 = np.zeros(N, dtype=np.complex_); psi0_2[A_site_start2] = 1;

psi1 = SolveSchrodingerGeneral(N,centres,funcs,params, tspan, nTimesteps, psi0_1)
psi2 = SolveSchrodingerGeneral(N,centres,funcs,params, tspan, nTimesteps, psi0_2)


"""plot"""
# normaliser = mpl.colors.Normalize(vmin=-1, vmax=1)
linthresh = 1e-2
normaliser=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
x_positions = np.linspace(0, nTimesteps, int(nOscillations/n_osc_divisions+1))
x_labels = list(range(0, nOscillations+1, n_osc_divisions))



#flip one
psi2 = np.flip(psi2, axis=0)

PlotPsi(psi1, x_positions, x_labels,  "Ramp",
      normaliser)
PlotPsi(psi2, x_positions, x_labels, "Ramp",
      normaliser)


#plot difference
PlotTwoPsi(psi1, psi2, x_positions, x_labels, "Ramp",
      normaliser)


""" Is there a diode? """

centre = centres[0]
shakeLen = len(centres)

psi1AboveBorder = psi1[:centre,:-1]
psi2AboveBorder = psi2[:centre,:-1]
psi1BelowBorder = psi1[centre+shakeLen:,:-1]
psi2BelowBorder = psi2[centre+shakeLen:,:-1]
psi1AtShake = psi1[centre:centre+shakeLen,:-1]
psi2AtShake = psi2[centre:centre+shakeLen,:-1]

psiDiffAboveBorder = np.abs(psi1AboveBorder)**2 - np.abs(psi2AboveBorder)**2
psiDiffBelowBorder = np.abs(psi1BelowBorder)**2 - np.abs(psi2BelowBorder)**2
psiDiffAtShake = np.abs(psi1AtShake)**2 - np.abs(psi2AtShake)**2
psiDiff =  np.abs(psi1[:,:-1])**2 - np.abs(psi2[:,:-1])**2
psiOverallDiffAboveBorder = np.sum(psiDiffAboveBorder, axis=0)
psiOverallDiffBelowBorder = np.sum(psiDiffBelowBorder, axis=0)
psiOverallDiff = np.sum(psiDiff, axis=0)
psiDiffAtShake = np.sum(psiDiffAtShake, axis=0)

fig, ax = plt.subplots(figsize = (12,8))
plt.plot(t_eval, psiOverallDiffAboveBorder)
plt.title("Psi Diff Above Border [:45]\n", y=1.02)
plt.show()


fig, ax = plt.subplots(figsize = (12,8))
plt.plot(t_eval, psiDiffAtShake)
plt.title("Psi Diff At Shake [45:47]", y=1.02)
plt.show()


fig, ax = plt.subplots(figsize = (12,8))
plt.plot(t_eval, psiOverallDiffBelowBorder)
plt.title("Psi Diff Below Border [47:]", y=1.06)
plt.show()

# fig, ax = plt.subplots(figsize = (12,8))
# plt.plot(t_eval, psiOverallDiff)
# plt.title("Psi Overall Diff\n"+form+", "+paramsString)
# plt.show()


#%%

"""Circle"""

"""General Params"""  
N = 5#182; 
rtol=1e-11
form = "Circle"

"""wave 1 parameters""" # for ssdf
A_site_start1 = 0
"""wave 2 params"""
A_site_start2 = 4


a = 35
omega1=9.1; omega2=2*omega1; omega3 = omega1
phi1=0; phi2=pi/2; phi2=pi/4
onsite1=0; onsite2=5; onsite3 = 10
T=2*pi/omega1
centres = [1,2,3]
funcs = [Cosine, Cosine, Cosine]

params = [[a, omega1, phi1, onsite1], [a, omega2, phi2, onsite2], [a, omega3, phi3, onsite3]]


t = np.linspace(0, 4*2*pi/omega1, 100)
shake1 = funcs[0](params[0], t)
shake2 = funcs[1](params[1], t)
shake3 = funcs[2](params[2], t)
plt.plot(t, shake1+shake2+shake3)
plt.plot(t, np.abs(shake1+shake2+shake3))
plt.ylabel("V")
plt.xlabel("t")
plt.show()

plt.plot(t, shake1, label="site 1")
plt.plot(t, shake2, label="site 2")
plt.plot(t, shake3, label="site 3")
plt.ylabel("V")
plt.xlabel("t")
plt.legend()
plt.show()




"""solver params"""
nOscillations = 30
#how many steps we want. NB, this means we will solve for nTimesteps+1 times (edges)
nTimesteps = nOscillations*100
nTimes = nTimesteps+1
n_osc_divisions = 2
tspan = (0,nOscillations*T)
t_eval = np.linspace(tspan[0], tspan[1], nTimesteps)


"""Solve"""
psi0_1 = np.zeros(N, dtype=np.complex_); psi0_1[A_site_start1] = 1;
psi0_2 = np.zeros(N, dtype=np.complex_); psi0_2[A_site_start2] = 1;

psi1 = SolveSchrodingerGeneral(N,centres,funcs,params, tspan, nTimesteps, psi0_1, circleBoundary = 1)
psi2 = SolveSchrodingerGeneral(N,centres,funcs,params, tspan, nTimesteps, psi0_2, circleBoundary = 1)

"""plot"""
normaliser = mpl.colors.Normalize(vmin=-1, vmax=1)
# linthresh = 1e-2
# normaliser=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
x_positions = np.linspace(0, nTimesteps, int(nOscillations/n_osc_divisions+1))
x_labels = list(range(0, nOscillations+1, n_osc_divisions))


#flip one
psi2 = np.flip(psi2, axis=0)

PlotPsi(psi1, x_positions, x_labels,  "Circle 1",
      normaliser)
PlotPsi(psi2, x_positions, x_labels, "Circle 2",
      normaliser)


#plot difference
# title = form+", "+hamiltonianString+"\n"+paramsString+r"$,\>\psi_{start site 1}=" +str(A_site_start1)+r",\> \psi_{start site 2} = " +str(A_site_start2)+r"$"
PlotTwoPsi(psi1, psi2, x_positions, x_labels, "Circle diff",
      normaliser)



#%%
"""Probability current"""



circleBoundary = 1
probCurrent1 = ProbCurrentofPsi(psi1, circleBoundary=circleBoundary)
probCurrent2 = ProbCurrentofPsi(psi2, circleBoundary=circleBoundary)


PlotProbCurrent(probCurrent1, x_positions, x_labels,  r"prob current $\psi_1$",
      normaliser)
PlotProbCurrent(probCurrent2, x_positions, x_labels,  r"prob current $\psi_2$",
      normaliser)

#difference between probability currents
PlotProbCurrent(probCurrent1 - probCurrent2, x_positions, x_labels, "prob current difference", normaliser)


""" Look at probability current overall """

# centre = centres[0]
# shakeLen = len(centres)

# currentAboveBorder1 = probCurrent1[:centre,:-1]
# currentAboveBorder2 = probCurrent2[:centre,:-1]
# currentBelowBorder1 = probCurrent1[centre+shakeLen:,:-1]
# currentBelowBorder2 = probCurrent2[centre+shakeLen:,:-1]
# currentAtShake1 = probCurrent1[centre:centre+shakeLen,:-1]
# currentAtShake2 = probCurrent2[centre:centre+shakeLen,:-1]

totalCurrent1 = np.sum(probCurrent1, axis=0)[:-1]
totalCurrent2 = np.sum(probCurrent2, axis=0)[:-1]
# totalCurrentAboveShake1 = np.sum(currentAboveBorder1,  axis=0)
# totalCurrentAboveShake2 = np.sum(currentAboveBorder2,  axis=0)
# totalCurrentBelowShake1 = np.sum(currentBelowBorder1,  axis=0)
# totalCurrentBelowShake2 = np.sum(currentBelowBorder2,  axis=0)
# totalCurrentAtShake1 = np.sum(currentAtShake1, axis=0)
# totalCurrentAtShake2 = np.sum(currentAtShake2, axis=0)

fig, ax = plt.subplots(figsize = (16,8))
plt.plot(t_eval/T, totalCurrent1, label=r"$\sum_x j(x,t)_1$")
plt.plot(t_eval/T, totalCurrent2, label=r"$\sum_x j(x,t)_2$")
plt.xlabel("t/T")
plt.legend()
plt.title("total probability current", y=1.02)
plt.show()


# fig, ax = plt.subplots(figsize = (12,8))
# plt.plot(t_eval/T, totalCurrentAboveShake1, label=r"$\sum_x j(x,t)_1$")
# plt.plot(t_eval/T, totalCurrentAboveShake2, label=r"$\sum_x j(x,t)_2$")
# plt.xlabel("t/T")
# plt.legend()
# plt.title("total probability current above shake [:45]", y=1.02)
# plt.show()


# fig, ax = plt.subplots(figsize = (12,8))
# plt.plot(t_eval/T, totalCurrentAtShake1, label=r"$\sum_x j(x,t)_1$")
# plt.plot(t_eval/T, totalCurrentAtShake2, label=r"$\sum_x j(x,t)_2$")
# plt.xlabel("t/T")
# plt.legend()
# plt.title("total probability current at shake [45:47]", y=1.02)
# plt.show()


# fig, ax = plt.subplots(figsize = (12,8))
# plt.plot(t_eval/T, totalCurrentBelowShake1, label=r"$\sum_x j(x,t)_1$")
# plt.plot(t_eval/T, totalCurrentBelowShake2, label=r"$\sum_x j(x,t)_2$")
# plt.xlabel("t/T")
# plt.legend()
# plt.title("total probability current below shake [47:]", y=1.02)
# plt.show()

#%%
"""for poaster"""


""""Global parameters"""
N = 91#182; 
centre = 30#90;
rtol=1e-11

#SS-p params
a = 35
omega = a /jn_zeros(0,1)[0]
phi = 0
T = 2*pi / omega
onsite = 0
form = "SS-p"; hamiltonianString="$H(t)=H_0 + a \> \hat{n}_b \cos (\omega t + \phi) $"; paramsString = r"$a="+str(a)+r", \omega = "+"{:.2f}".format(omega)+", \phi = "+PhiString(phi)+r"$"


form = "SS-p"; 
hamiltonianString="$H(t)=H_0 + a \> \hat{n}_b \cos (\omega t + \phi) $"; 
paramsString = r"$a="+str(a)+r", \omega = "+"{:.2f}".format(omega)+", \phi = "+PhiString(phi)+r"$"


"""wave 1 parameters""" # for ssdf
A_site_start1 = 40#85;


# plot potential
t = np.linspace(0, 4*2*pi/omega, 100)
shake = a*cos(omega*t + phi) 



"""solver params"""
nOscillations = 30
#how many steps we want. NB, this means we will solve for nTimesteps+1 times (edges)
nTimesteps = nOscillations*100
n_osc_divisions = 2
tspan = (0,nOscillations*T)
t_eval = np.linspace(tspan[0], tspan[1], nTimesteps)


psi0_1 = np.zeros(N, dtype=np.complex_); psi0_1[A_site_start1] = 1;
psi = SolveSchrodinger(form, rtol, N, centre, a, omega, phi, 
                                  tspan, nTimesteps, psi0_1, onsite)


"""plot"""


# normaliser = mpl.colors.Normalize(vmin=0, vmax=1)
linthresh = 1e-2
normaliser=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
# normaliser=mpl.colors.LogNorm(vmin=0, vmax=1.0, base=10)
x_positions = np.linspace(0, nTimesteps, int(nOscillations/n_osc_divisions+1))
x_labels = list(range(0, nOscillations+1, n_osc_divisions))



"""
Plot Matter Wave
"""

mpl.rcParams.update({
      'mathtext.fontset':'stix'
      })

apply = [lambda x: np.abs(x)**2]
labels = [r'$|\psi(t)|^2$']


cmapcol = 'PuOr' #PiYG_r
cmap= mpl.cm.get_cmap(cmapcol)

sz = 4
fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                        figsize=(sz*1.7,sz))

for i, f in enumerate(apply):
    ax.matshow(f(psi), interpolation='none', cmap=cmap, norm=normaliser, aspect='auto')
    ax.set_title(labels[i],  fontfamily='STIXGeneral')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xticks(x_positions)
    ax.set_xlabel('t/T', fontfamily='STIXGeneral')
    ax.set_xticklabels(x_labels)
    for side in ["bottom", "top", "left", "right"]:
        ax.spines[side].set_visible(False)
    if i == 0:
        ax.set_ylabel('site', fontfamily='STIXGeneral')

cax = plt.axes([1.03, 0.1, 0.03, 0.8])
fig.colorbar(plt.cm.ScalarMappable(cmap=cmapcol, norm=normaliser), cax=cax)
# fig.suptitle(title, y = 1.2,  fontfamily='STIXGeneral')
paper = "/Users/Georgia Nixon/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Paper/Figures/"
plt.savefig(paper + "reflection.png", dpi=700)

plt.show()