# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:57:59 2021

@author: Georgia Nixon
"""
import numpy as np
from scipy.integrate import quad, dblquad
from numpy import sin, cos, pi, exp
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl

size=16
params = {
            'legend.fontsize': size*0.75,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size,
          'font.family': 'STIXGeneral',
#          'axes.titlepad': 25,
          'mathtext.fontset': 'stix'
          }


mpl.rcParams.update(params)
# plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
# plt.rcParams['grid.color'] = "0.9" # grid axis colour


CB91_Blue = 'darkblue'#'#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
red = "#FC4445"

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
               CB91_Purple,
                # CB91_Violet,
                'dodgerblue',
                'slategrey']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

def convert_complex(s):
    return np.complex(s.replace('i', 'j').replace('*I', 'j').replace('*^', 'e'))

def phistring(phi):
    if phi == 0:
        return "0"
    else:
        return  r'$\pi /$' + str(int(1/(phi/pi)))
    
def XIntegrandSimplifiedOld(y,x,omega, A, phi):
    fac = A/omega
    return 2*sin(fac*(sin(omega*x + phi) - sin(omega*y + phi)))


def XIntegrandSimplifiedT1(y,x,omega, A, phi):
    theta2 = y
    theta1 = x
    fac = A/omega
    t1 = (2/omega**2)*sin(fac*sin(theta1))*cos(fac*sin(theta2))
    t2 = -(2/omega**2)*cos(fac*sin(theta1))*sin(fac*sin(theta2))
    return t1

def XIntegrandSimplifiedT2(y,x,omega, A, phi):
    theta2 = y
    theta1 = x
    fac = A/omega
    t1 = (2/omega**2)*sin(fac*sin(theta1))*cos(fac*sin(theta2))
    t2 = -(2/omega**2)*cos(fac*sin(theta1))*sin(fac*sin(theta2))
    return t2

def XIntegrandSimplified(y,x,omega, A, phi):
    theta2 = y
    theta1 = x
    fac = A/omega
    t1 = (2/omega**2)*sin(fac*sin(theta1))*cos(fac*sin(theta2))
    t2 = -(2/omega**2)*cos(fac*sin(theta1))*sin(fac*sin(theta2))
    return t1 + t2


#def StarIntegrand(y,x,omega,A,phi):
#    fac = 1j*A/omega
#    chi1 = omega*x + phi
#    chi2 = omega*y + phi
#    return 0.5*(exp(fac*sin(chi1)) - exp(fac*sin(chi2)))
#
#def SquareIntegrand(y,x,omega,A,phi):
#    fac = 1j*A/omega
#    chi1 = omega*x + phi
#    chi2 = omega*y + phi
#    return 0.5*(exp(-fac*sin(chi2)) - exp(-fac*sin(chi1)))


def GetResults(Func, phis, omegas, nres):

    ResultsPhis = np.zeros((len(phis), nres), dtype=np.complex128)
    for j, phi in enumerate(phis):
        Results = np.zeros(nres, dtype=np.complex128)
        for i, omega in enumerate(omegas):
            
            #limits if we are working with t
#            mins = 0
#            max1 = 2*pi/omega
            #limits if we are working with theta
            mins = phi
            max1 = phi + 2*pi

            F = lambda y, x: Func(y,x,omega, A, phi)
            r, _ = dblquad(F, mins, max1, lambda x: mins, lambda x: x)
            Results[i] = r
            
        ResultsPhis[j] = Results
    return ResultsPhis
    
#%%
    


A = 35

nres = 160
omegas = np.linspace(3.7, 60, nres, endpoint=True)
phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]

xResultsPhis0 = GetResults(XIntegrandSimplifiedT1, phis, omegas, nres)

fig, ax = plt.subplots()
for j, phi in enumerate(phis):
    ax.plot(omegas, np.real(xResultsPhis0[j]), label=r"$\phi=$"+phistring(phi) + "  2")
fig.legend()
plt.show()

xResultsPhis1 = GetResults(XIntegrandSimplifiedT2, phis, omegas, nres)

fig, ax = plt.subplots()
for j, phi in enumerate(phis):
    ax.plot(omegas, np.real(xResultsPhis1[j]), label=r"$\phi=$"+phistring(phi) + "  2")
fig.legend()
plt.show()

xResultsPhis2 = GetResults(XIntegrandSimplified, phis, omegas, nres)

fig, ax = plt.subplots()
for j, phi in enumerate(phis):
    ax.plot(omegas, np.real(xResultsPhis2[j]), label=r"$\phi=$"+phistring(phi) + "  2")
fig.legend()
plt.show()

#%%

from mpmath import nsum, inf
from mpmath import exp as mpmathexp
from scipy.special import jv

a = nsum(lambda k: jv(2*k + 1, 1)/(2*k + 1), [0, inf])

#%%

sh = '/Users/Georgia Nixon/Code/MBQD/floquet-simulations/'
df = pd.read_csv(sh+'data/analysis-G.csv', 
                  index_col=False, 
                  converters={
                       'hopping': convert_complex,
                                'onsite':convert_complex,
                                'next onsite':convert_complex,
                                'NNN':convert_complex, 
                              'NNN overtop':convert_complex,
                                              })

fig, ax = plt.subplots()
for nc, phi in enumerate(phis):
    df_plot = df[(df['form']=='SS-p')&
                                     (df['N']==51)&
                                          (df['a']==A) &
                                          (df['phi']==phi)
                                          ]
    df_plot = df_plot.sort_values(by=['omega'])
    ax.plot(df_plot['omega'], np.real(df_plot['onsite'].values), label=r"$\phi=$"+phistring(phi))
fig.legend()  
plt.show()

#%%
from scipy.integrate import quad
from scipy.special import jv

phi = 0
As = np.linspace(0.001, 100, 1000)
phis = np.linspace(0, 2*pi, 60)
x = np.linspace(-pi, 2*pi, 60)
nevens = np.linspace(2, 52, 26, endpoint=True)
nodds = np.linspace(1, 51, 26, endpoint=True)

A = 100
results2 = [quad(lambda x: sin(n*x)*sin(A*sin(x)), 0,pi) for n in nevens]
plt.plot(nevens, results2)
plt.show()
#%%

A = 1
results = [quad(lambda x: sin(A*sin(x))*x, phi, 2*pi+phi) for phi in phis]
plt.plot(phis, results)
plt.show()

results1 =  [quad(lambda x: sin(A*sin(x))*x, 0, 2*pi) for A in As]
plt.plot(As, results1)
plt.show()
#%%

results3 = [quad(lambda x: sin(2*x)*sin(A*sin(x)), 0,pi) for A in As]
results3 = np.array([jv(0, A) for A in As])

#%%
fig, ax = plt.subplots(figsize=(10,10))
jvplot = np.zeros(len(As))
for n in np.linspace(1, 51, 6, endpoint=True):
    plt.plot(As, jv(n,As), label=str(n))
    jvplot = jvplot + jv(n, As)
  
plt.legend()
plt.show()

plt.plot(As, jvplot)
plt.show()


    
#plt.plot(As, jv(0,As))
#plt.show()
