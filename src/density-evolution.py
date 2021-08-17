
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
from hamiltonians import SolveSchrodinger, SolveSchrodingerGeneral
from hamiltonians import  PhiString

import matplotlib as mpl
import seaborn as sns
from scipy.special import jv, jn_zeros
from fractions import Fraction

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/"

size=25
params = {
            'legend.fontsize': size*0.75,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size,
          'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
          'axes.grid':False,
          'font.family' : 'STIXGeneral',
          'mathtext.fontset':'stix'
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
                            figsize=(sz*len(apply),sz))
    
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
    

    
def PhiStringNum(phi):
    if phi == 0:
        return "0"
    elif phi == "phi":
        return r"$\phi$" 
    else:
        return  r"\pi /" + str(int(1/(phi/pi)))
    


#%%
"""
Compare two matter waves
"""


""""Global parameters"""
N = 92#182; 
centre = 45#90;
rtol=1e-11

#SS-p params
# a = 35
# omega = 10#a /jn_zeros(0,1)[0]
# phi = 0
# T = 2*pi / omega
<<<<<<< HEAD
# form = "SS-p"; hamiltonianString="$H(t)=H_0 + a \> \hat{n}_b \cos (\omega t + \phi) $"; paramsString = r"$a="+str(a)+r", \omega = "+"{:.2f}".format(omega)+", \phi = "+PhiString(phi)+r"$"

=======
# form = "SS-p"; 
# hamiltonianString="$H(t)=H_0 + a \> \hat{n}_b \cos (\omega t + \phi) $"; 
# paramsString = r"$a="+str(a)+r", \omega = "+"{:.2f}".format(omega)+", \phi = "+PhiString(phi)+r"$"
>>>>>>> 13ed4195f523f7e1b2a7650aac205da698cc3b37


# form = "DS-p"; 
# hamiltonianString = (r"$H(t)=H_0 + \hat{n}_b [a \> \cos (\omega_1 t + \phi_1) + s_1]  + "
#                      +r"\hat{n}_{b+1} [a \> \cos (\omega_2 t + \phi_2) + s_2]$"); 
# paramsString = (r"$a="+str(a)+", "+r"\omega_1="+"{:.2f}".format(omega1)+
#                 ", \omega_2 = "+str(omegaMultiplier)+" \omega_1, \phi_1 = "+PhiString(phi1)
#                 +", \phi_2 = \phi_1 + \pi/2, s_1 = " + str(onsite1)+r", s_2 = "+ str(onsite2)
#                 + r", N = "+str(N)+", b = "+str(centre)+"$ ")
# form = "SSDF-p"; hamiltonianString = "$H(t)=H_0 + a \> \hat{n}_b [\cos (\omega_1 t + \phi_1)  +  \cos (\omega_2 t + \phi_2)]$"; paramsString = r"$a=$"+str(a)+", "+r"$\omega_1="+ "{:.2f}".format(omega1)+", \omega_2 = "+str(omegaMultiplier)+" \omega_1, \phi_1 ="+PhiString(phi1)+", \phi_2 = \phi_1 + \pi/2, N = "+str(N)+", b = "+str(centre)+"$ "



#DS-p params
form = "DS-p"
a = 35
phi1=0;
<<<<<<< HEAD
phiOffset2=pi/4
phiOffset3 = pi/4
phi2=phi1+phiOffset2
phi3 = phi1+phiOffset3
phi=[phi1,phi2, phi3]
onsite1 = 0
onsite2 = 10
onsite3 = 20
onsite = [onsite1, onsite2, onsite3]
omega1= 10#a/jn_zeros(0,1)[0]
omegaMultiplier2=2
omegaMultiplier3 = 3
omega2=omega1*omegaMultiplier2
omega3 = omega1*omegaMultiplier3
omega=[omega1,omega2, omega3]
T=2*pi/min(omega)
form = "TS-p"; 
=======
phiOffset=pi/2
phi2=phi1+phiOffset
onsite1 = 0
onsite2 = 20
onsite = [onsite1, onsite2]
omega1= 9.1#a/jn_zeros(0,1)[0]
omegaMultiplier=2
omega2=omega1*omegaMultiplier
phi=[phi1,phi2]
omega=[omega1,omega2]
T=2*pi/omega1
form = "DS-p"; 
>>>>>>> 13ed4195f523f7e1b2a7650aac205da698cc3b37
hamiltonianString = (r"$H(t)=H_0 + \hat{n}_b [a \> \cos (\omega_1 t + \phi_1) + s_1]  + "
                      +r"\hat{n}_{b+1} [a \> \cos (\omega_2 t + \phi_2) + s_2]$"); 
paramsString = (r"$a="+str(a)+", "+r"\omega_1="+"{:.2f}".format(omega1)+
                ", \omega_2 = "+str(omegaMultiplier)+" \omega_1, \phi_1 = "+PhiString(phi1)
                +", \phi_2 = \phi_1 + \pi/2, s_1 = " + str(onsite1)+r", s_2 = "+ str(onsite2)
                + r", N = "+str(N)+", b = "+str(centre)+"$ ")

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
#                      +r"\hat{n}_{b+1} [a \> \cos (\omega_2 t + \phi_2) + s_2] + "
#                      +r"\hat{n}_{b+2} [a \> \cos (\omega_3 t + \phi_3) + s_3]  $"); 
# paramsString = (r"$a="+str(a)+", "+r"\omega_1="+"{:.2f}".format(omega1)+
#                 ", \omega_2 = "+str(omegaMultiplier2)+" \omega_1, \omega_3 = "+str(omegaMultiplier3)+"\omega_1, "
#                 +r"\phi_1 = "+PhiString(phi1) +", \phi_2 = \phi_1 + "+PhiString(phiOffset2) + r", \phi_3 = \phi_1 + "+PhiString(phiOffset3)
#                 +r", s_1 = " + str(onsite1)+r", s_2 = "+ str(onsite2) +r", s_3 = "+str(onsite3)
#                 + r", N = "+str(N)+", b = "+str(centre)+"$ ")

"""wave 1 parameters""" # for ssdf
A_site_start1 = 40#85;
"""wave 2 params"""
A_site_start2 = 51#96;



<<<<<<< HEAD
# plot potential
t = np.linspace(0, 4*2*pi/omega1, 100)
shake1 = a*cos(omega1*t + phi1) + onsite1
shake2 = a*cos(omega2*t + phi2) + onsite2
shake3 = a*cos(omega3*t + phi3) + onsite3
plt.plot(t, shake1+shake2+shake3, label="V(t)")
plt.plot(t, np.abs(shake1+shake2+shake3), label = "abs(V(t))")
=======
t = np.linspace(0, 4*2*pi/omega1, 100)
shake1 = a*cos(omega1*t + phi1) + onsite1
shake2 = a*cos(omega2*t + phi2) + onsite2
# shake3 = a*cos(omega3*t + phi3) + onsite3
plt.plot(t, shake1+shake2)
plt.plot(t, np.abs(shake1+shake2))
>>>>>>> 13ed4195f523f7e1b2a7650aac205da698cc3b37
plt.ylabel("V")
plt.xlabel("t")
plt.legend()
plt.show()





<<<<<<< HEAD



=======
>>>>>>> 13ed4195f523f7e1b2a7650aac205da698cc3b37
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
<<<<<<< HEAD
psi1BelowBorder = psi1[centre+3:,:-1]
psi2BelowBorder = psi2[centre+3:,:-1]
psi1AtShake = psi1[centre:centre+3,:-1]
psi2AtShake = psi2[centre:centre+3,:-1]
=======
psi1BelowBorder = psi1[centre+shakeLen:,:-1]
psi2BelowBorder = psi2[centre+shakeLen:,:-1]
psi1AtShake = psi1[centre:centre+shakeLen,:-1]
psi2AtShake = psi2[centre:centre+shakeLen,:-1]
>>>>>>> 13ed4195f523f7e1b2a7650aac205da698cc3b37

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
Old titles
"""


# title = (r"$|\psi_1> = |\psi_{\phi="+PhiStringNum(phi1)+"}(t)>, \>"
#          +"|\psi_2> = |\psi_{\phi="+PhiStringNum(phi2)+"}(t)>$"
#          +"\n"
#          +r"evolution via G, "
#          + r"given $H(t)=H_0 + "+str(a)
#          +r"\cos (" + "{:.2f}".format(omega)+ r"t + \phi"
#          + r") |"+str(centre)+r"><"+str(centre) +r"|,"
#          +r" \quad  |\psi (t=0)> = |"+str(A_site_start)+r">$"
#         )  



# title = (r"$|\psi (t)>$"+  "\n"
#          # +r"evolution via G, "
#          # + r"given $H(t)=H_0 + 35 \cos (" + "{:.2f}".format(omega1)
#          #     + r"t" + PhiString(phi1) 
#          #     + r") |"+str(centre)+r"><"+str(centre) +r"|,"
#          +form+", "+r"$a_1=$"+str(a)
#              +", "+r"$\omega_1=$"+"{:.2f}".format(omega1)
#               +", "+r"$\omega_2=$"+"{:.2f}".format(omega2)
#               +", "+r"$\phi_1=$"+ PhiStringNum(phi1)
#               +", "+r"$\phi_2=$"+ PhiStringNum(phi2)
#               +", "+"b="+str(centre)
              
#               +r", $\quad |\psi (t=0)> = |"+str(A_site_start)+r">$"
         
         
#          ) 

# title = (r"$|\psi (t)>$"+  "\n"
#          +r"evolution via G, "
#          + r"given $H(t)=H_0 + 35 \cos (" + "{:.2f}".format(omega)
#              + r"t" + PhiString(phi2) 
#              + r") |"+str(centre)+r"><"+str(centre) +r"|,"
#              +r" \quad  |\psi (t=0)> = |"+str(A_site_start)+r">$"
#          ) 


    
#%%


"""
Homogeneous expansion
"""

def H_0(N, centre, el):
    H = np.zeros((N, N), dtype=np.complex128)
    H = H + np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)
    H[centre][centre-1] = -exp(1j*el)
    H[centre-1][centre] = -exp(-1j*el)
    H[centre+1][centre] = -exp(-1j*el)
    H[centre][centre+1]= -exp(1j*el)
    assert(np.all(0 == (np.conj(H.T) -H)))
    return H



def H_0_T(N, centre, t):
    omega = 0.01*2*pi #want omega = 2\pi / J and J is 1
    H = np.zeros((N, N), dtype=np.complex128)
    H = H + np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)
    H[centre][centre-1] = -exp(1j*omega*t)
    H[centre-1][centre] = -exp(-1j*omega*t)
    H[centre+1][centre] = -exp(-1j*omega*t)
    H[centre][centre+1]= -exp(1j*omega*t)
    assert(np.all(0 == (np.conj(H.T) -H)))
    return H

# no energy offset at all
def F_0(t, psi, N, centre, el):
    return -1j*np.dot(H_0(N, centre, el), psi)

def F_0_T(t, psi, N, centre):
    return -1j*np.dot(H_0_T(N, centre, t), psi)

from scipy.integrate import solve_ivp

def SolveSchrodingerH(form, N, centre, rtol, tspan, nTimesteps, psi0, el=0):
    # points to calculate the matter wave at
    t_eval = np.linspace(tspan[0], tspan[1], nTimesteps+1, endpoint=True)
    
    if form == "H0":
        sol = solve_ivp(lambda t,psi: F_0(t, psi, 
                               N, centre, el), 
                t_span=tspan, y0=psi0, rtol=rtol, 
                atol=rtol, t_eval=t_eval,
                method='RK45')
        
    elif form == "H0T":
        sol = solve_ivp(lambda t,psi: F_0_T(t, psi, 
                               N, centre), 
                t_span=tspan, y0=psi0, rtol=rtol, 
                atol=rtol, t_eval=t_eval,
                method='RK45')
    
    sol=sol.y
        
    return sol


form = "H0T"
N = 91; 
centre = 45;
rtol=1e-11
T = 1

nOscillations = 30
#how many steps we want. NB, this means we will solve for nTimesteps+1 times (edges)
nTimesteps = nOscillations*100
n_osc_divisions = 2
tspan = (0,nOscillations*T)
t_eval = np.linspace(tspan[0], tspan[1], nTimesteps)

aSiteStart = 45
psi0 = np.zeros(N, dtype=np.complex_); psi0[aSiteStart] = 1;

psi1 = SolveSchrodingerH("H0", N, centre, rtol, tspan, nTimesteps, psi0, el=0)

psi2 = SolveSchrodingerH("H0T", N, centre, rtol, tspan, nTimesteps, psi0)
        
# normaliser = mpl.colors.Normalize(vmin=-1, vmax=1)
linthresh = 1e-3
normaliser=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)

x_positions = np.linspace(0, nTimesteps, int(nOscillations/n_osc_divisions+1))
x_labels = list(range(0, nOscillations+1, n_osc_divisions))

PlotPsi(psi1, x_positions, x_labels, r"Hamiltonian with hoppings around site 45 moving in complex plane with frequency $\omega = 2 \pi$"+"\n"+r"atom starts at site 45",
      normaliser)

PlotPsi(psi2, x_positions, x_labels,   r"Hamiltonian with hoppings around site 45 moving in complex plane with frequency $\omega = 2 \pi$"+"\n"+r"atom starts at site 45",
      normaliser)

# PlotPsi(psi1-psi2, x_positions, x_labels,  "title",
#       normaliser)

#plot difference
PlotTwoPsi(psi1, psi2, x_positions, x_labels, r"$|\psi_1>; \phi = 0, |\psi_2>; \phi = \pi$"+"\n"+"b="+str(centre)+", atom starts at site "+str(aSiteStart),
      normaliser)

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
omega1= 20; omega2=2*omega1; omega3=3*omega1
phi1=0; phi2=pi/4; phi3=pi/2
onsite1=0; onsite2=10; onsite3=20
T=2*pi/omega1
centres = [45,46, 47]

params = [[a, omega1, phi1, onsite1], [a, omega2, phi2, onsite2], [a, omega3, phi3, onsite3]]
funcs = [Blip, Blip, Blip]


t = np.linspace(0, 4*2*pi/omega1, 100)
shake1 = funcs[0](params[0], t)
shake2 = funcs[1](params[1], t)
shake3 = funcs[2](params[2], t)
plt.plot(t, shake1+shake2+shake3)
plt.plot(t, np.abs(shake1+shake2+shake3))
plt.ylabel("V")
plt.xlabel("t")
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
# title = form+", "+hamiltonianString+"\n"+paramsString+r"$,\>\psi_{start site 1}=" +str(A_site_start1)+r",\> \psi_{start site 2} = " +str(A_site_start2)+r"$"
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




