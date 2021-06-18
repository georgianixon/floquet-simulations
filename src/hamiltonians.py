# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:25:20 2020

@author: Georgia
"""

from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.special import jv, jn_zeros
import pandas as pd


from scipy.linalg import eig as eig
from scipy.linalg import expm



"""
Time dependent linear energy offset
"""

def HT_Linear(N, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)  
    for i in range(N):
        matrix[i][i] = a*i*cos(omega*t + phi)
    return matrix



"""
Create one site cosine modulated energy offset hamiltonian
Centre indexed from 0
"""
def HT_SS(N, centre, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega*t + phi)
    return matrix

def HT_DS(N, centre, a1, a2, omega1, omega2, t, phi1, phi2):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a1*cos(omega1*t + phi1)
    matrix[centre+1][centre+1] = a2*cos(omega2*t + phi2)
    return matrix


"""
No energy offset
"""
def H_0(N):
    return np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          


"""
Functions to Solve Schrodinger eq
"""


# one site cosine 
def F_SS(t, psi, N, centre, a, omega, phi):
    return -1j*np.dot(HT_SS(N, centre, a, omega, t, phi),psi)

def F_DS(t, psi, N, centre, a1, a2, omega1, omega2, phi1, phi2):
    return -1j*np.dot(HT_DS(N, centre, a1, a2, omega1, omega2, t, phi1, phi2),psi)

# linear moving potential
def F_Linear(t, psi, N, a, omega, phi):
    return -1j*np.dot(HT_Linear(N, a, omega, t, phi), psi)

# no energy offset at all
def F_0(t, psi, N):
    return -1j*np.dot(H_0(N), psi)


def F_HF(t, psi, HF):
    return -1j*np.dot(HF, psi)




def roundcomplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j

def solve_schrodinger(form, rtol, N, centre, a, omega, phi, tspan, n_timesteps, psi0):
    """
    solve time dependent schrodinger eq given initial conditions psi0, over
    time tspan, for Hamiltonian signified by 'form'
    """
    
    if form=="DS-p":
        a1 = a[0]
        a2 = a[1]
        omega1 = omega[0]
        omega2 = omega[1]
        phi1 = phi[0]
        phi2 = phi[1]
        
    t_eval = np.linspace(tspan[0], tspan[1], n_timesteps+1)
    
    if form == 'SS-p':
        sol = solve_ivp(lambda t,psi: F_SS(t, psi, 
                           N, centre,
                             a,
                             omega, phi), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
        
    elif form == 'DS-p':
        sol = solve_ivp(lambda t,psi: F_DS(t, psi, 
                           N, centre,
                             a1, a2,
                             omega1, omega2,
                             phi1, phi2), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
        
        
    elif form =='linear' or form == "linear-p":
        sol= solve_ivp(lambda t,psi: F_Linear(t, psi, N, a, omega, phi), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
        
        
    elif form == 'numericalG-SS-p':
        _, HF = create_HF('SS-p', rtol, N, centre, a, None, None, phi, omega)
        HFr = roundcomplex(HF, 5)
        assert(np.all(0 == (HFr - np.conj(HFr.T))))
        evals, evecs= eig(HFr)
        coeffs =  np.dot(np.conj(evecs.T), psi0)
        sol = [np.dot(evecs, coeffs*exp(-1j*evals*t)) for t in t_eval]
        sol = np.vstack(sol).T
        
        
    return sol




def create_HF(form, rtol, N, centre, a,phi, omega): 

    
    
    assert(form in ['linear', "linear-p", "SS-p", "DS-p"])
    if form == "DS-p":
        T = 2*pi/omega[0]
    else:
        T=2*pi/omega
    tspan = (0,T)
    UT = np.zeros([N,N], dtype=np.complex_)
    
    for A_site_start in range(N):
    #    print(A_site_start)
        psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
        sol = solve_schrodinger(form, rtol, N, centre, a, omega, phi, tspan, 100, psi0)
        UT[:,A_site_start]=sol[:,-1] 
    
    # print(time.time()-start, 'seconds.')
    
    evals_U, evecs = eig(UT)
    evals_H = 1j / T *log(evals_U)
    
    HF = np.zeros([N,N], dtype=np.complex_)
    for i in range(N):
        term = evals_H[i]*np.outer(evecs[:,i], np.conj(evecs[:,i]))
        HF = HF+term
        
    # print('   ',time.time()-start, 's')
    HFr = roundcomplex(HF, 5)
    assert(np.all(0 == (HFr - np.conj(HFr.T))))

    return UT, HF




def hoppingHF(N, centre, a, omega, phi):
    HF =  np.zeros([N,N], dtype=np.complex_)
    HF = HF + np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1) 
    entry = -exp(-1j*a*sin(phi)/omega)*jv(0, a/omega)
    HF[centre][centre+1] = entry # the term we collect for hoppping
    HF[centre][centre-1] = entry
    HF[centre+1][centre] = np.conj(entry)
    HF[centre-1][centre] = np.conj(entry)
    HFr = roundcomplex(HF, 7)
    assert(np.all(0 == (HFr - np.conj(HFr.T))))
    return HF, entry


#%%

def roundcomplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j


def getevalsandevecs(HF):
    evals, evecs = eig(HF)
    idx = np.real(evals).argsort()
    evals = evals[idx]
    evecs = evecs[:,idx]
    # evecsR = roundcomplex(evecs, 5)
    evalsR = roundcomplex(evals, 5)
    assert(np.all(np.imag(evalsR)==0))
    evals = np.real(evals)
    return evals, evecs

def formatcomplex(num, dp):
    return ("{:."+str(dp)+"f}").format(num.real) + " + " + ("{:."+str(dp)+"f}").format(num.imag) + "i"

def plotevecs(evecs, N, func, colour, title, ypos=0.934):
    sz = 3
    num = 7
    fig, ax = plt.subplots(nrows = num, ncols = num, sharex=True,
                           sharey=True,
                           figsize=(sz*num,sz*num))

    for i in range(num):
        for j in range(num):
            evec1 = evecs[:,num*i + j]
    
            ax[i,j].plot(range(N), func(evec1), color=colour) 

    fig.suptitle(title, y=ypos)
    plt.show()
    
def plotevals(evals, N, title, ypos=1.06):
    assert(np.all(0 == np.imag(evals)))
    sz = 6
    fig, ax = plt.subplots(figsize=(sz*1.4,sz))
    ax.plot(range(N), np.real(evals), 'x', color="#304050")
    fig.suptitle(title, y=ypos)
    plt.show()
    
    
#%%

#order evecs0 to
def OrderEvecs(evecs0, N):
    """
    Make first nonzero element of evecs real and positive
    """
    evecs0_R = roundcomplex(evecs0, 5)
    for vec in range(N):
        #index of first non zero element of this vector
        firstnonzero = (evecs0_R[:,vec]!=0).argmax()
        #make this value real and positive, so arg(this value)=2*pi*n for integer n
        angle = np.angle(evecs0[firstnonzero,vec])
        evecs0[:,vec] = evecs0[:,vec]*exp(-1j*angle)
    return evecs0


def AlignEvecs(evecs0, evecsP, N):
    evecs0_R = roundcomplex(evecs0, 5)
    evecsP_R = roundcomplex(evecsP, 5)
    
    #flip if needed
    for vec in range(N):
        #if one is negative of the other, for rounded evecs, flip one,
        #could both start at zero
        if np.all(evecs0_R[:,vec]==-evecsP_R[:,vec]):
            evecsP[:,vec] = -evecsP[:,vec]
            #redefine rounded evecsP
            evecsP_R[:,vec] = roundcomplex(evecsP[:,vec], 5)
        # else, make them start at the same value
        elif evecs0_R[0,vec] != evecsP_R[0,vec]:
            frac = evecs0[0,vec] / evecsP[0,vec]
            evecsP[:,vec] = evecsP[:,vec] * frac
            #redefine evecsP_R
            evecsP_R[:,vec] = roundcomplex(evecsP[:,vec], 5)
            
    return evecsP