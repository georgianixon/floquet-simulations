# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:01:15 2020

@author: Georgia
"""

import matplotlib as mpl
place="Georgia Nixon"
from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("//wsl$/Ubuntu-20.04/home/georgianixon/projects/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
from hamiltonians import CreateHF, CreateHFGeneral, HT_SS, hoppingHF, GetEvalsAndEvecsGen
from hamiltonians import Cosine, RemoveWannierGauge, DoubleCosine, OnsiteOnly
from scipy.special import jn_zeros, jv
from fractions import Fraction 
# from hamiltonians import Cosine, PhiString
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh


posterLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Posters/202205 DAMOP/Final/"
# def RemoveWannierGauge(matrix, c, N):
#     phase = np.angle(matrix[c-1,c])
#     gaugeMatrix = np.identity(N, dtype=np.complex128)
#     gaugeMatrix[c,c] = np.exp(-1j*phase)
#     matrix = np.matmul(np.matmul(np.conj(gaugeMatrix), matrix), gaugeMatrix)
#     return matrix


def Plot():
    size=12
    params = {
                'legend.fontsize': size,
    #          'figure.figsize': (20,8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size*0.9,
              'ytick.labelsize': size*0.9,
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
              'font.family' : 'STIXGeneral',#"sans-serif",#"Arial"#
              "font.sans-serif":"stix",#"Arial",
               'mathtext.fontset':"stix"#"Arial"#'stix'
              }
    
    mpl.rcParams.update(params)

    

Plot()

# def GetPhiOffsetReal(phi1Real, phiTOffset, omegaMultiplier):
#     if omegaMultiplier == 1.5:
#         piEquivA = 1/4
#         piEquivB = 1/6
#     phi1T = phi1Real/pi*piEquivA
#     phi2T = phi1T+phiTOffset
#     phi2Real = phi2T/piEquivB*pi
#     return phi2Real

def GetPhiOffset(time1, timeOffset, omega1, omega2):
    time2 = time1+timeOffset
    
    omegaT = np.gcd(round(100*omega1), round(100*omega2))/100
    totalT = 2*pi/omegaT
    
    phi1 = time1*omega1*totalT
    phi2 = time2*omega2*totalT
    
    return phi1, phi2, totalT


# def RampHalf(params, t): # ramp
#     a = params[0]
#     omega = params[1]
#     phi = params[2]
#     onsite = params[3]
    
#     nHalfCycle = np.floor(t*omega/pi + phi/pi)
#     y = (a*omega*t/pi + a*phi/pi - nHalfCycle*a)*((nHalfCycle + 1 ) % 2) + onsite
#     return y

# def RampGen(params, t): # ramp
#     a = params[0]
#     omega = params[1]
#     phi = params[2]
#     theta = params[4] 
#     onsite = params[3]
    
#     nCycles = np.floor(t*omega/2/pi)
#     tCycle = t - nCycles*2*pi/omega
#     multiplier = 1
#     if tCycle < phi/omega:
#         multiplier = 0
#     elif tCycle > (phi + theta)/omega:
#         multiplier = 0
    

#     subtract_height = 2*a*(pi)/theta*nCycles
#     y = (a*omega*t/theta - a*phi/theta - subtract_height)*multiplier + onsite
#     return y


def RampGen(params, t): # ramp
    a = params[0]
    omega = params[1]
    phi = params[2]
    theta = params[4] 
    onsite = params[3]

    
    nCycles = np.floor(t*omega/2/pi)
    tCycle = t - nCycles*2*pi/omega
    
    multiplier_pre_phi = (np.sign(tCycle - phi/omega)%3)%2
    multiplier_post_theta =  (np.sign(-tCycle + phi/omega + theta/omega)%3)%2

    subtract_height = 2*a*(pi)/theta*nCycles
    y = (a*omega*t/theta - a*phi/theta - subtract_height)*multiplier_pre_phi*multiplier_post_theta + onsite
    return y


def ComputeAValsFromRequiredGradients(gradients):
    N = len(gradients)
    xvals = np.zeros(N)
    xzero = 2.4048
    for i, y in enumerate(gradients):
        if y > 0:
            sol = minimize_scalar(lambda x: np.abs(jv(0,x) - y),
                              bounds = (0,xzero),
                              method="bounded")
            xvals[i] = sol.x
        elif y < 0:
            sol = minimize_scalar(lambda x: np.abs(jv(0,x) - y),
                              bounds = (xzero, 3.8316),
                              method="bounded")
            xvals[i] = sol.x
    return xvals

#get A_vals
def GetAValsFromBesselXVals(bessel_x_vals, omega, addition_type = "accumulative", constant_shift=""):
    """make returning A Vals jump around 0 if accumulative = False
    Let A vals accumulate if accumulative = True
    Choose constant_shift to be one of 'zero centre', 'positive', or none 
    """
    
    if (addition_type != "accumulative") and (addition_type != "+2,-2") and (addition_type != "alternating"):
        raise TypeError("right type please")
        
    A_diff_vals = bessel_x_vals*omega
    A_vals = [0]
    for i, diff in enumerate(A_diff_vals):
        if addition_type == "accumulative":
            A_vals.append(A_vals[i] + diff)
        elif addition_type == "+2,-2":
            if (i %4 == 0) or (i%4 == 1):
                A_vals.append(A_vals[i] + diff)
            else:
                A_vals.append(A_vals[i] - diff)
        elif addition_type == "alternating":
            if i%2 == 0:
                A_vals.append(A_vals[i] + diff)
            else:
                A_vals.append(A_vals[i] - diff) 
    A_vals = np.array(A_vals)
    if constant_shift=="positive":
        A_vals_min = np.min(A_vals)
        A_vals = A_vals - A_vals_min
    elif constant_shift == "zero centre":
        A_vals_spread = np.max(A_vals) - np.min(A_vals)
        A_vals = A_vals - np.max(A_vals) + A_vals_spread/2
    return A_vals

# def GetAValsFromBesselXVals(bessel_x_vals, omega, accumulative = False, positive=True):
#     """make returning A Vals jump around 0 if accumulative = False
#     Let A vals accumulate if accumulative = True
#     """
#     A_diff_vals = bessel_x_vals*omega
#     A_vals = [0]
#     for i, diff in enumerate(A_diff_vals):
#         if (i % 4== 0) or (i % 4 == 1):
#             A_vals.append(A_vals[i] + diff)
#         else:
#             if accumulative:   
#                 A_vals.append(A_vals[i] + diff)
#             else:
#                 A_vals.append(A_vals[i] - diff) 
#     A_vals = np.array(A_vals)
#     if positive==True:
#         A_vals_min = np.min(A_vals)
#         A_vals = A_vals - A_vals_min
#     return A_vals

def Plot3G(HF):
    absMax = np.max([np.abs(np.min(np.real(HF))),
                    np.abs(np.max(np.real(HF))),
                    np.abs(np.min(np.imag(HF))),
                    np.abs(np.max(np.imag(HF)))])

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    norm = mpl.colors.Normalize(vmin=-absMax, vmax=absMax)
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
    # 

    '''abs real imag'''

    apply = [
             np.abs, 
             np.real, np.imag]
    labels = [
              r'$\mathrm{Abs}\{G_{n,m}\}$', 
              r'$\mathrm{Re}\{G_{n,m}\}$',
              r'$\mathrm{Imag}\{G_{n,m}\}$'
              ]

    sz = 8
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                           figsize=(sz,sz/2))

    for n1, f in enumerate(apply):
        pcm = ax[n1].matshow(f(HF), interpolation='none', cmap='PuOr',  norm=norm)
        ax[n1].set_title(labels[n1])
        ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[n1].set_xlabel('m')

    ax[0].set_ylabel('n', rotation=0, labelpad=10)

    # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    cax = plt.axes([1.03, 0.2, 0.03, 0.6])
    # fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
    fig.colorbar(pcm, cax=cax)
    plt.show()
    
def PlotRG(HF):
    absMax = np.max([np.abs(np.min(np.real(HF))),
                    np.abs(np.max(np.real(HF))),
                    np.abs(np.min(np.imag(HF))),
                    np.abs(np.max(np.imag(HF)))])

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    norm = mpl.colors.Normalize(vmin=-absMax, vmax=absMax)
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
    # 

    '''abs real imag'''


    sz = 3
    fig, ax = plt.subplots(constrained_layout=True, 
                           figsize=(sz,sz))
    pcm = ax.matshow(np.real(HF), interpolation='none', cmap='PuOr',  norm=norm)
    ax.set_title( r'$\mathrm{Re}\{H_{n,m}\}$')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xlabel('m')

    ax.set_ylabel('n', rotation=0, labelpad=10)

    # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    cax = plt.axes([0.95, 0.16, 0.06, 0.74])
    # fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
    fig.colorbar(pcm, cax=cax)
    plt.show()

#%%

"""
Plot HF---------------

Set form = ... and other parameters
Plot the Real, Imag and Abs parts of the floquet Hamiltonian
Using createHF general
"""


# to try and get black hole
# _, HF = CreateHFGeneral(3,
#                          [0,1,2],
#                          [Cosine]*3,
#                          [[,10,0,0] for i in range(10)],
#                          2*pi/10,
#                          0
#                          )
 
        
# common params
# N=8;   rtol=1e-11

# form="SS-p"
# phi1 = pi/4
# phiOffset = pi/2
# phi2 = phi1+phiOffset
# phis=[phi1, phi2];
# omega1 = a1/jn_zeros(0,1)[0]
# omegaMultiplier=2
# omega2 = omega1*omegaMultiplier
# omegas = [omega1, omega2]

# N = 12
# form = "SSHModel"
# centre = np.nan
# a = [16.917322691940058, 38.315948852085626]
# omega1 = 10; omega = [omega1, omega1]
# phi = [0, 0]
# onsite =[0,0]# [0.095430,-0.1211175387]
# rtol = 1e-6
# _, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)

# PlotRG(HF)
#SSH
N = 8
centres = np.array(range(N))
funcs = [Cosine, Cosine, Cosine, Cosine]*2
a1 = 16.917322691940058; a2 = -50; a3 = 38.315948852085626; a4 = -10
omega1 = 25; omega2 = omega1; omega3 = omega1; omega4 = omega3; T = 2*pi/omega1
phi1 = 0; phi2 = 0; phi3 = 0; phi4 = 0
onsite0 = -0.06886628325878923; onsite1 =0.0697854749060796; onsite2 = 0.068;  onsite3=0;onsite4 = -0.068
# params = [onsite0, [a1, omega1, phi1, onsite1], [(a2, a3), (omega2, omega3), (phi2, phi3), onsite2], [a4, omega4, phi4, onsite4]]*2
params = [ [a1, omega1, phi1, onsite1], [a2, omega2, phi2, onsite2], [a3, omega3, phi3, onsite3], [a4, omega4, phi4, onsite4]]*2
circleBoundary = 0
_, HF = CreateHFGeneral(N, centres, funcs, params, T, circleBoundary)
 

PlotRG(HF)


#triangle
# N = 3
# centres= [1,2]
# funcs = [Cosine, Cosine]
# a1 = 15; a2 = 15;
# omega0 = 10
# T = 2*pi/omega0
# alpha = 1; beta = 2
# omega1 = alpha*omega0; omega2 = beta*omega0
# phi1 = 0; phi2 = 2*pi/3
# onsite = 0
# paramss = [[a1, omega1, phi1, onsite], [a2, omega2, phi2, onsite]]
# circleBoundary = 1


# a = 35
# omegaMultiplier = 1.5
# omega1 = 6.8; omega2=omegaMultiplier*omega1
# time1 = 4/8
# timeOffset  = 2/3 # units of total time period
# phi1, phi2, T  = GetPhiOffset(time1, timeOffset, omega1, omega2)



#full lattice shake
# form = 'linear'
# centre = np.nan
# a = 10
# omega = 10
# phi = pi/3
# onsite = np.nan
# _, HF = CreateHF(form, rtol, N, centre, a, omega, phi, onsite)


# _, HF = CreateHFGeneral(N, centres, funcs, paramss, T, circleBoundary)
# 

# for site in range(N):
#     HF = RemoveWannierGauge(HF, site, N)




# HFevals, HFevecs = GetEvalsAndEvecsGen(HF)
# HFabs = np.copy(HF)
# HFabs[0,2] = -np.abs(HFabs[0,2])
# HFabs[2,0] = -np.abs(HFabs[2,0])
# # HFabs = np.abs(HF)
# HFevalsabs, _ = GetEvalsAndEvecsGen(HFabs)


# sz = 7
# fig, ax = plt.subplots(figsize=(sz/2,sz))
# ax.plot([0]*len(HFevals), HFevals, 'o')
# ax.plot([1]*len(HFevals), HFevalsabs, 'o')
# # ax.axes.xaxis.set_visible(False)
# # ax.set_title("evals")
# ax.set_xticks([0, 1])
# ax.set_xticklabels([r'$\mathrm{evals}_{real}$', r'$\mathrm{evals}_{abs}$'])
# ax.set_xlim([-0.3, 1.4])
# plt.show()

#%%

a = 10
omega0 = 10
alpha = 1
beta = 3
phi2_frac = 0.8

tshift_frac = 0.1
phi3_frac = beta*(tshift_frac + phi2_frac/alpha)
phi3rel_frac = phi3_frac - phi2_frac

tshiftfrac1 = phi3rel_frac/beta + phi2_frac/beta  - phi2_frac/alpha
print(tshift_frac, tshiftfrac1)

t = np.linspace(0,2*pi/omega0,100)
two = 10*cos(alpha*omega0*t + phi2_frac*pi)
three = a*cos(beta*omega0*t + phi2_frac*pi + phi3rel_frac*pi)

plt.plot(t, two, label="two")
plt.plot(t, three, label="three")
plt.legend()
plt.show()



#%%


a = 10; omega = 2*pi; phi = 0; onsite = 0; theta = pi

t = np.linspace(0,2*pi/omega,1000)
y = RampGen([a, omega, phi, onsite, theta],t) 

plt.plot(t,y,'.')
#%%

""" Get A vals from specified tunnelling values
We use this for generating the linear gradient and the black hole stuff or also SSH using addition_type +2,-2"""

# get the A vals to get the right gradient
Ndiffs = 8
ymin = jv(0, 3.8316)
omega = 25
xzero = 2.4048
# gradients = np.linspace(-ymin, ymin, Ndiffs) # for linear
# gradients = np.array([ymin, -ymin]*(int(Ndiffs/2)))  # for SSH
gradients = np.array([0.4, -0.1, 0.1]*4)
# gradients = np.array([0.9, 0.4]*int(Ndiffs/2)) # for SSH not negative, makes A's lower
xvals = ComputeAValsFromRequiredGradients(gradients)
A_vals = GetAValsFromBesselXVals(xvals, omega, addition_type="alternating", constant_shift="zero centre") # get actual shaking values

#oscilating A_vals
# A_vals = np.array([i%2 for i in range(Ndiffs +1)])*10

N= len(A_vals)
fig, ax = plt.subplots(figsize=(3,2))
ax.plot(range(N), A_vals, '.')
ax.set_ylabel(r"$A$", rotation = 0)
ax.set_xlabel(r"$i$")
ax.set_xticks(np.arange(0,N,10))
plt.show()


print([round(i, 2) for i in A_vals])
_, HF = CreateHFGeneral(N,
                          [int(i) for i in list(np.linspace(0,N-1,N))],
                          [Cosine]*(N),
                          [[i,omega,0,0] for i in A_vals], #a, omega, phi onsite
                          2*pi/omega,
                          0
                          )
#offset onsites
# for i in range(N):
#     HF[i,i]=0
    
# for i in range(N-2):
#     HF[i, i+2] = 0
#     HF[i+2, i] = 0
Plot3G(HF)


PlotRG(HF)
  
  
ymax = 0.43
# plot gradient
fig, ax = plt.subplots(figsize=(3,2))
JNN = [np.real(HF[i,i+1]) for i in range(N-1)]
ax.plot(range(N-1), JNN, '.', label = r"$J_i$")
# plt.plot(range(N-1), -gradients, label=r"linear gradient")
ax.set_xlabel("i")
ax.set_ylabel(r"$J_{i, i+1}$")
ax.set_ylim([-ymax, ymax])
ax.set_xticks(np.arange(0,N-1,10))
# plt.legend()
plt.show()


# plot gradient 2
JNNN = [np.real(HF[i,i+2]) for i in range(N-2)]
fig, ax = plt.subplots(figsize=(3,2))
ax.plot(range(N-2), JNNN, '.')

ax.set_ylabel(r"$J_{i, i+2}$")
ax.set_ylim([-ymax, ymax])
ax.set_xlabel("i")
ax.set_xticks(np.arange(0,N-1,10))
plt.show()


onsite = [np.real(HF[i,i]) for i in range(N)]
fig, ax = plt.subplots(figsize=(3,2))
ax.plot(range(N), onsite, '.')
ax.set_ylabel(r"$H_{i, i}$")
ax.set_ylim([-ymax, ymax])
ax.set_xlabel("i")
ax.set_xticks(np.arange(0,N,10))
plt.show()

JNNN_spread = np.abs(np.max(JNNN)) + np.abs(np.min(JNNN))
JNN_spread = np.abs(np.max(JNN)) + np.abs(np.min(JNN))

print("JNN/JNNN:", "{:.3f}".format(JNN_spread/JNNN_spread), "\t JNN spread:", "{:.3f}".format(JNN_spread), 
      "\t JNNN spread:", "{:.3f}.".format(JNNN_spread), "\t A_val max:", "{:.3f}".format(np.max(A_vals)))

HF = np.real(HF)
#%%

def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

HFdf = pd.DataFrame(np.real(HF))
HFdf.to_csv("/Users/GeorgiaNixon/OneDrive - Riverlane/PhD/LinearMetric/Hamiltonian_csvs_for_aydin/H_w=25_N=101_accumulativeA_withOnsites.csv", 
            index=False, header=False)


#%%
 
# N = 5
# j1  = 0.9
# j2 = 0.5

# HF = np.diag(np.array([j1,j2]*(int(N-1))),-1)+np.diag(np.array([j1,j2]*(int(N-1))),1)

# Plot3G(HF)

HFevals, HFevecs = eigh(HF)


for i in range(2,7):#range(len(HFevecs)):
    fig, ax = plt.subplots()    
    ax.plot(range(len(HFevecs[:,0])), HFevecs[:,i])
    plt.show()

fig, ax = plt.subplots()
ax.plot(range(len(HFevals)), HFevals, 'x')
plt.show()



#%%

# def bmatrix(a):
#     """Returns a LaTeX bmatrix

#     :a: numpy array
#     :returns: LaTeX bmatrix as a string
#     """
#     if len(a.shape) > 2:
#         raise ValueError('bmatrix can at most display two dimensions')
#     lines = str(a).replace('[', '').replace(']\n', 'endline').replace("\n", "").replace("endline", "\n").replace("]]", "").splitlines()
#     rv = [r'\begin{bmatrix}']
#     rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
#     rv +=  [r'\end{bmatrix}']
#     return '\n'.join(rv)


# def tocsv(a):
#     """Returns a LaTeX bmatrix

#     :a: numpy array
#     :returns: LaTeX bmatrix as a string
#     """
#     if len(a.shape) > 2:
#         raise ValueError('bmatrix can at most display two dimensions')
#     lines = str(a).replace('[', '').replace(']\n', 'endline').replace("\n", "").replace("endline", "\n").replace("]]", "").splitlines()
#     rv = [r'\begin{bmatrix}']
#     rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
#     rv +=  [r'\end{bmatrix}']
#     return '\n'.join(rv)


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

HFdf = pd.DataFrame(np.real(HF))
HFdf.to_csv("/Users/GeorgiaNixon/OneDrive - Riverlane/PhD/LinearMetric/Hamiltonian_csvs_for_aydin/H_w=2_N=101_accumulativeA_withOnsites.csv", index=False, header=False)

# print(tocsv(signif(HF, 4)))
# print(bmatrix(signif(HF, 4)))
#%%

"""Get HF from general """
# _, HF = CreateHFGeneral(10,
#                           [0,1,2,3,4,5,6,7,8,9],
#                           [Cosine]*10,
#                           [[15,10,0,i/10] for i in range(10)], #a, omega, phi onsite
#                           2*pi/10,
#                           0
#                           )

a = 0

onsite = 0
N = Ndiffs + 1

N = 12
_, HF = CreateHFGeneral(N,
                          [int(i) for i in np.linspace(0,N -1, N)],
                          [Cosine]*N,
                          [[10,omega,i*pi/4,onsite] for i in range(1,N+1)],
                          2*pi/omega,
                          0
                          )


# for i in range(N):
#     HF[i][i] = 0
    
    
absMax = np.max([np.abs(np.min(np.real(HF))),
                np.abs(np.max(np.real(HF))),
                np.abs(np.min(np.imag(HF))),
                np.abs(np.max(np.imag(HF)))])

latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Analytics/"
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
norm = mpl.colors.Normalize(vmin=-absMax, vmax=absMax)
# linthresh = 1e-1
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
# 

'''abs real imag'''

apply = [
         np.abs, 
         np.real, np.imag]
labels = [
          r'$\mathrm{Abs}\{G_{n,m}\}$', 
          r'$\mathrm{Re}\{G_{n,m}\}$',
          r'$\mathrm{Imag}\{G_{n,m}\}$'
          ]

sz = 8
fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))

for n1, f in enumerate(apply):
    pcm = ax[n1].matshow(f(HF), interpolation='none', cmap='PuOr',  norm=norm)
    ax[n1].set_title(labels[n1])
    ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[n1].set_xlabel('m')

ax[0].set_ylabel('n', rotation=0, labelpad=10)

# cax = plt.axes([1.03, 0.1, 0.03, 0.8])
cax = plt.axes([1.03, 0.2, 0.03, 0.6])
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)
    
latexLoc = "C:/Users/Georgia/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Analytics/"
# fig.savefig(latexLoc+"G,Triangle,Example,Gauged.pdf", 
#         format='pdf', bbox_inches='tight')
plt.show()




#%%

# Hawking Temperature calculation
import math
N = 40
alpha = 10
d = 0.1
nh = N/2
ymin = jv(0, 3.8316)
kn = np.zeros(N)
for i in range(N):
    kn[i] = alpha*math.tanh(d*(i- nh- 0.5))/4/d
knmax = np.max(np.abs(kn))
gradients = kn/knmax*ymin

plt.plot(range(N), -gradients)
plt.show()

#get A vals to get the right gradient

xzero = 2.4048
omega = 8

xvals = ComputeAValsFromRequiredGradients(gradients) # get bessel x values
A_vals = GetAValsFromBesselXVals(xvals, omega, accumulative=True) # get actual shaking values
N= len(A_vals)
fig, ax = plt.subplots()
ax.plot(range(N), A_vals, '.')
ax.set_ylabel(r"$A$")
ax.set_xticks(np.arange(0,N,2))
ax.set_xlabel(r"$i$")
plt.show()


print([round(i, 2) for i in A_vals])
_, HF = CreateHFGeneral(N,
                          [int(i) for i in list(np.linspace(0,N-1, N))],
                          [Cosine]*(N),
                          [[i,omega,0,0] for i in A_vals], #a, omega, phi onsite
                          2*pi/omega,
                          0
                          )
#offset onsites
for i in range(N):
    HF[i,i]=0
PlotG(HF)
# plot gradient
fig, ax = plt.subplots()
y = [np.round(np.real(HF[i,i+1]), 3) for i in range(N-1)]
ax.plot(range(N-1), y, '.', label = r"$J_i$")
plt.plot(range(N-1), -gradients, label=r"$10 \> \tanh(0.1*(i- nh- 0.5))/(4*0.1)$")
ax.set_xlabel("i")
ax.set_ylabel(r"$J_{i, i+1}$")
ax.set_ylim([-0.41, 0.41])
ax.set_xticks(np.arange(0,N-1,2))
plt.legend()
plt.show()


# plot gradient 2
y = [np.round(np.real(HF[i,i+2]), 3) for i in range(N-2)]
fig, ax = plt.subplots()
ax.plot(range(N-2), y, '.')
ax.set_ylabel(r"$J_{i, i+2}$")
ax.set_ylim([-0.41, 0.41])
ax.set_xlabel("i")
ax.set_xticks(np.arange(0,N-2,2))
plt.show()



#%%

# fig for paper


norm = mpl.colors.Normalize(vmin=0, vmax=1)
# linthresh = 1e-1
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
cmapcol = "Purples"#'PuOr' #PiYG_r
cmap= mpl.cm.get_cmap(cmapcol)


'''abs real imag'''

apply = [np.abs]
labels = [r'$\mathrm{Abs}\{G_{n,m}\}$']

sz = 2.1
fig, ax = plt.subplots( constrained_layout=True, figsize=(sz,sz))


pcm = ax.matshow(np.abs(HF), interpolation='none', cmap='Purples',  norm=norm)
ax.set_title(labels[0])
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
  labeltop=False)  
ax.set_xlabel('m')

ax.set_ylabel('n', rotation=0, labelpad=10)

    
    
cax = plt.axes([0.92, 0.12, 0.06, 0.8])
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)



paper = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Paper/Figures/"
# fig.savefig(paper+'G-SSHModel-small.pdf', format='pdf', bbox_inches='tight')
plt.show()

#%%

#  Real part of Hamiltonian 

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# linthresh = 1e-2
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
# 

paperLoc= "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Writing/Local Modulations/"

'''abs real imag'''
Plot()
cmap = "RdBu"#"PuOr"

sz = 2
fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, 
                       figsize=(sz,sz))

pcm = ax.matshow(np.real(HF), interpolation='none', cmap=cmap,  norm=norm)
ax.set_title(r'$\mathrm{Real}\{H_{n,m}^{\mathrm{eff}}\}$')
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
  labeltop=False)  
ax.set_xlabel('m')

ax.set_ylabel('n', rotation=0, labelpad=10)

ax.set_ylabel('n', rotation=0, labelpad=14)
    
# cax = plt.axes([0.97, 0.145, 0.08, 0.78]) #size = 6
# cax = plt.axes([0.95, 0.145, 0.045, 0.76]) # size=3
cax = plt.axes([0.92, 0.22, 0.06, 0.645])
# cax = plt.axes([0.9, 0.265, 0.08, 0.59]) 
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)
fig.savefig(paperLoc+'G-SSH-Real.pdf', format='pdf', bbox_inches='tight')
plt.show()

#%%

# Abs part of Hamiltonian 

norm = mpl.colors.Normalize(vmin=0, vmax=1)
# linthresh = 1e-1
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
# 

'''abs real imag'''



sz = 2
fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, 
                       figsize=(sz,sz))

pcm = ax.matshow(np.abs(HF), interpolation='none', cmap='Purples',  norm=norm)
ax.set_title(r'$\mathrm{Abs}\{G_{n,m}\}$')
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
  labeltop=False)  
ax.set_xlabel('m')

ax.set_ylabel('n', rotation=0, labelpad=10)

ax.set_ylabel('n', rotation=0, labelpad=14)
    
cax = plt.axes([0.97, 0.145, 0.08, 0.78])
# fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
fig.colorbar(pcm, cax=cax)
# fig.savefig(paper+'G-Triangle-Real.pdf', format='pdf', bbox_inches='tight')
plt.show()





