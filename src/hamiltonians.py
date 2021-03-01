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
    return 1j*np.dot(HT_SS(N, centre, a, omega, t, phi),psi)

# linear moving potential
def F_Linear(t, psi, N, a, omega, phi):
    return 1j*np.dot(HT_Linear(N, a, omega, t, phi), psi)

# no energy offset at all
def F_0(t, psi, N):
    return 1j*np.dot(H_0(N), psi)


def F_HF(t, psi, HF):
    return 1j*np.dot(HF, psi)




def roundcomplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j

def solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi, tspan, n_timesteps, psi0):
    """
    solve time dependent schrodinger eq given initial conditions psi0, over
    time tspan, for Hamiltonian signified by 'form'
    """
    
    t_eval = np.linspace(tspan[0], tspan[1], n_timesteps+1)
    
    if form == 'SS-p':
        sol= solve_ivp(lambda t,psi: F_SS(t, psi, 
                           N, centre,
                             a,
                             omega, phi), 
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




def create_HF(form, rtol, N, centre, a,b, c,phi, omega): 
    
    assert(form in ['linear', "linear-p", "SS-p"])
    T=2*pi/omega
    tspan = (0,T)
    UT = np.zeros([N,N], dtype=np.complex_)
    
    for A_site_start in range(N):
    #    print(A_site_start)
        psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
        sol = solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi, tspan, 100, psi0)
        UT[:,A_site_start]=sol[:,-1] 
    
    # print(time.time()-start, 'seconds.')
    
    evals_U, evecs = eig(UT)
    evals_H = 1j / T *log(evals_U)
    
    HF = np.zeros([N,N], dtype=np.complex_)
    for i in range(N):
        term = evals_H[i]*np.outer(evecs[:,i], np.conj(evecs[:,i]))
        HF = HF+term
        
    # print('   ',time.time()-start, 's')
    
    return UT, HF






