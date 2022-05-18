# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:25:20 2020

@author: Georgia
"""

from numpy import exp, cos, log
import numpy as np

from scipy.integrate import solve_ivp

from scipy.linalg import eig 
from scipy.linalg import eigh 

def GetEvalsAndEvecsGen(HF, fix_gauge=1):
    """
    Get e-vals and e-vecs of Hamiltonian HF.
    Order Evals and correspoinding evecs by smallest eval first.
    Set the gauge for each evec; choosing the first non-zero element to be real and positive.
    Note that the gauge may be changed later by multiplying any vec arbitrarily by a phase. 
    """
    
    # check if hermitian
    if np.all(np.round(np.conj(HF.T),15)==np.round(HF,15)):
        evals, evecs = eigh(HF) # evals are automatically real
    
    else:
        # print("matrix not hermitian")
        #order by evals, also order corresponding evecs
        evals, evecs = eig(HF)
        idx = np.real(evals).argsort()
        evals = evals[idx]
        evecs = evecs[:,idx]
    
    # all evecs have a gauge 
    # make first element of evecs real and positive
    if fix_gauge:
        for vec in range(np.size(HF[0])):
            
            # Find first element of the first eigenvector that is not zero
            firstNonZero = (evecs[:,vec]!=0).argmax()
            #find the conjugate phase of this element
            conjugatePhase = np.conj(evecs[firstNonZero,vec])/np.abs(evecs[firstNonZero,vec])
            #multiply all elements by the conjugate phase
            evecs[:,vec] = conjugatePhase*evecs[:,vec]

    # check that the evals are real
    
    return evals, evecs

    # if np.all((np.round(np.imag(evals),10) == 0)) == True:
    #     return np.real(evals), evecs
    # else:
    #     # x =  evals[np.argsort(np.imag(evals))[0]]
    #     # print('evals are imaginary! e.g.', f"{x:.3}")
    #     return evals, evecs

def H0(N):
    return np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)      


def HT_General(N, centres, funcs, paramss, circleBoundary, t):
    H = H0(N)
    # do we want periodic boundary conditions?
    if circleBoundary:
        H[0,-1]=-1
        H[-1,0]=-1
    
    for i in range(len(centres)):
        func = funcs[i]
        H[centres[i],centres[i]] = func(paramss[i], t)
    return H



def F_General(t, psi, N, centre, func, params, circleBoundary):
    H = HT_General(N, centre, func, params, circleBoundary, t)
    return -1j*np.dot(H, psi)


from fractions import Fraction
def ConvertFraction(s):
    """
    For retrieving fractions from csv's
    """
    maxDenom=24
    return Fraction(s).limit_denominator(maxDenom).__float__()

def ConvertComplex(s):
    """
    For retrieving complex numbers from csv's
    """
    return np.complex128(s.replace('i', 'j').replace('*I', 'j').replace('*^', 'e'))

def RoundComplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j


def SolveSchrodingerGeneral(N,centre,func,params, tspan, nTimesteps, psi0, circleBoundary = 0):
    
    rtol=1e-11
    # points to calculate the matter wave at
    t_eval = np.linspace(tspan[0], tspan[1], nTimesteps+1, endpoint=True)
    sol = solve_ivp(lambda t,psi: F_General(t, psi, 
                                          N, centre, func, params, circleBoundary), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
    sol=sol.y
    return sol




def CreateHFGeneralLoopA(N, centre, func, params, T, circleBoundary): 
    """
    If there is an error, return nans (to keep loop going)
    """
    tspan = (0,T)
    UT = np.zeros([N,N], dtype=np.complex_)
    nTimesteps = 100
    
    for A_site_start in range(N):
    #    print(A_site_start)
        psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
        sol = SolveSchrodingerGeneral(N, centre, func, params, tspan, nTimesteps, psi0, circleBoundary=circleBoundary)
        UT[:,A_site_start]=sol[:,-1] 
    
    # print(time.time()-start, 'seconds.')
    
    # evals_U, evecs = eig(UT)
    evals_U, evecs = GetEvalsAndEvecsGen(UT) #evals can be imaginary
    evals_H = 1j / T *log(evals_U)
    
    HF = np.zeros([N,N], dtype=np.complex_)
    for i in range(N):
        term = evals_H[i]*np.outer(evecs[:,i], np.conj(evecs[:,i]))
        HF = HF+term
        
    # print('   ',time.time()-start, 's')
    HFr = RoundComplex(HF, 4)
    if np.all(0 == (HFr - np.conj(HFr.T))):
        return UT, HF
    else:
        return np.nan, np.nan
    




"""Usual Cos shake"""
def Cosine(params, t):
    a = params[0]
    omega = params[1]
    phi = params[2]
    onsite = params[3]
    y = a*cos(omega*t + phi)+ onsite
    return y 


def RemoveWannierGauge(matrix, c, N):
    phase = np.angle(matrix[c-1,c])
    phase = phase - np.pi #because it should be np.pi (ie negative)
    gaugeMatrix = np.identity(N, dtype=np.complex128)
    gaugeMatrix[c,c] = np.exp(-1j*phase)
    matrix = np.matmul(np.matmul(np.conj(gaugeMatrix), matrix), gaugeMatrix)
    return matrix
        