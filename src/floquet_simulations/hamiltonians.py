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


from scipy.linalg import eig 
from scipy.linalg import eigh 
from scipy.linalg import expm

def GetEvalsAndEvecsGen(HF, fix_gauge=1):
    """
    Get e-vals and e-vecs of Hamiltonian HF.
    Order Evals and correspoinding evecs by smallest eval first.
    Set the gauge for each evec; choosing the first non-zero element to be real and positive.
    Note that the gauge may be changed later by multiplying any vec arbitrarily by a phase. 
    """
    
    # check if hermitian
    if np.all(np.round(np.conj(HF.T),12)==np.round(HF,12)):
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



    
def getevalsandevecs(HF):
    """
    for some reason, this is different to GetEvalsAndEvecs..
    """
    #order by evals, also order corresponding evecs
    evals, evecs = eig(HF)
    idx = np.real(evals).argsort()
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    #make first element of evecs real and positive
    for vec in range(np.size(HF[0])):
        # phi = np.angle(evecs[0,vec])
        # evecs[:,vec] = exp(-1j*phi)*evecs[:,vec]
#        evecs[:,vec] = np.conj(evecs[0,vec])/np.abs(evecs[0,vec])*evecs[:,vec]
        
        #nurs normalisation
        evecs[:,vec] = np.conj(evecs[1,vec])/np.abs(evecs[1,vec])*evecs[:,vec]
    return evals, evecs

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
def HT_SS(N, centre, a, omega, phi, onsite, t):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega*t + phi) + onsite
    return matrix

def HT_DS(N, centre, a, omega1, omega2, phi1, phi2, onsite1, onsite2, t):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega1*t + phi1) + onsite1
    matrix[centre+1][centre+1] = a*cos(omega2*t + phi2) + onsite2
    return matrix

def HT_TS(N, centre, a, omega, phi, onsite, t):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega[0]*t + phi[0]) + onsite[0]
    matrix[centre+1][centre+1] = a*cos(omega[1]*t + phi[1]) + onsite[1]
    matrix[centre+2][centre+2] = a*cos(omega[2]*t + phi[2]) + onsite[2]
    return matrix

def HT_SSDF(N, centre, a, omega1, omega2, phi1, phi2, onsite,  t):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)
    matrix[centre][centre] = a*cos(omega1*t + phi1) + a*cos(omega2*t + phi2) + onsite
    return matrix

def HT_Circle(N, centre, a, omega,  phi, onsite,  t):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)
    matrix[0,-1] = -1
    matrix[-1,0] = -1
    matrix[centre][centre] = a*cos(omega*t + phi)
    return matrix


def HT_StepFunc(N, centre, a, omega,  phi, onsite,  t):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)
    for i in range(centre, N):
        matrix[i, i] = a*cos(omega*t + phi)
    return matrix

def HT_StepFuncGen(N, centres, aas, omegas,  phis, onsites,  t):
    H = H0(N)
    for centre, a, omega, phi, onsite in zip(centres, aas, omegas, phis, onsites):
        for i in range(centre, N):
            H[i, i] = H[i,i] + a*cos(omega*t + phi) + onsite
    return H


def HT_SSHModel(N, aas, omegas,  phis, onsites,  t):
    
    assert(N/4 == int(N/4))
    nCycles = int(N/4)
    
    a1 = aas[0]
    a2 = aas[1]
    omega1 = omegas[0]
    omega2 = omegas[1]
    phi1 = phis[0]
    phi2 = phis[1]
    onsite1 = onsites[0]
    onsite2 = onsites[1]
    
    H = H0(N)
    
    for cycle in range(nCycles):
        i = cycle*4 + 1
        H[i,i] = a1*cos(omega1*t + phi1) + onsite1
        H[i+1, i+1] = a1*cos(omega1*t + phi1) + onsite1 + a2*cos(omega2*t + phi2) + onsite2
        H[i+2, i+2] = a2*cos(omega2*t + phi2) + onsite2
        
    return H



"""
No energy offset
"""
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




"""Moving phases"""

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

"""
Time independent Hamiltonians
"""

def H0_PhasesNNHop(N, centres, els):
    H = np.zeros((N, N), dtype=np.complex128)
    H = H + H0(N)
    for centre, el in zip(centres, els):
        H[centre][centre-1] = -exp(1j*el)
        H[centre-1][centre] = -exp(-1j*el)
        # H[centre+1][centre] = -exp(-1j*el)
        # H[centre][centre+1]= -exp(1j*el)
    assert(np.all(0 == (np.conj(H.T) -H)))
    return H

def H0_PhasesLongRangeHop(N, centre, p0, p1, p2, p3, p4=0):
    H = np.zeros((N,N), dtype = np.complex128)
    H = H + H0(N)
    for i in range(centre-2, centre+3):
        H[i,i] =  p0 # this cannot be complex otherwise it will not be hermitian
    for i in range(centre-2, centre+2):
        if not np.isnan(p1):
            #hopping
            H[i, i+1] = -exp(1j*p1)
            H[i+1,i] = -exp(-1j*p1)
    for i in range(centre-2, centre+1):
        if not np.isnan(p2):
            #NNN hopping
            H[i, i+2] = -exp(1j*p2)
            H[i+2,i] = -exp(-1j*p2)
    for i in range(centre-2, centre):
        if not np.isnan(p3):
            H[i, i+3] = -exp(1j*p3)
            H[i+3,i] = -exp(-1j*p3)
    for i in range(centre-2, centre-1):
        if not np.isnan(p4):
            H[i, i+4] = -exp(1j*p4)
            H[i+4,i] = -exp(-1j*p4)
    return H



def H0_Triangle(J1, J2):
    H = np.zeros((3,3), dtype=np.complex128)
    H[0,1] =-J2
    H[1,2] =-np.conj(J2)
    H[2,0] =-J1
    H[1,0] =-np.conj(-J2)
    H[2,1] =-J2
    H[0,2] =-np.conj(-J1)
    return H
    
def H0_DipoleTrap(N, centre, dipoleFac):
    H = np.zeros((N,N), dtype = np.complex128)
    H = H + H0(N)
    for i in range(N):
        H[i,i] =  dipoleFac*np.abs(centre-i)**2 # this cannot be complex otherwise it will not be hermitian
    return H

def H0_DipoleTrapwPhases(N, centre, dipoleFac, p1, p2, p3, p4):
    H = np.zeros((N,N), dtype = np.complex128)
    H = H + H0(N)
    for i in range(N):
        H[i,i] =  dipoleFac*np.abs(centre-i)**2 # this cannot be complex otherwise it will not be hermitian
    
    # add phases
    for i in range(centre-2, centre+2):
        if not np.isnan(p1):
            #hopping
            H[i, i+1] = -exp(1j*p1)
            H[i+1,i] = -exp(-1j*p1)
    for i in range(centre-2, centre+1):
        if not np.isnan(p2):
            #NNN hopping
            H[i, i+2] = -exp(1j*p2)
            H[i+2,i] = -exp(-1j*p2)
    for i in range(centre-2, centre):
        if not np.isnan(p3):
            H[i, i+3] = -exp(1j*p3)
            H[i+3,i] = -exp(-1j*p3)
    for i in range(centre-2, centre-1):
        if not np.isnan(p4):
            H[i, i+4] = -exp(1j*p4)
            H[i+4,i] = -exp(-1j*p4)
    return H








"""
Functions to Solve Schrodinger eq
"""


# one site cosine 
def F_SS(t, psi, N, centre, a, omega, phi, onsite):
    H = HT_SS(N, centre, a, omega, phi, onsite, t)
    return -1j*np.dot(H, psi)

def F_DS(t, psi, N, centre, a, omega1, omega2, phi1, phi2, onsite1, onsite2):
    H = HT_DS(N, centre, a, omega1, omega2, phi1, phi2, onsite1, onsite2, t)
    return -1j*np.dot(H, psi)

def F_SSDF(t, psi, N, centre, a, omega1, omega2, phi1, phi2, onsite):
    H = HT_SSDF(N, centre, a, omega1, omega2, phi1, phi2, onsite, t)
    return -1j*np.dot(H, psi)

def F_Circle(t, psi, N, centre, a, omega,phi,  onsite):
    H = HT_Circle(N, centre, a, omega,  phi,  onsite, t)
    return -1j*np.dot(H, psi)

def F_StepFunc(t, psi, N, centre, a, omega, phi, onsite):
    H = HT_StepFunc(N, centre, a, omega, phi, onsite, t)
    return -1j*np.dot(H, psi)

def F_StepFuncGen(t, psi, N, centres, aas, omegas, phis, onsites):
    H = HT_StepFuncGen(N, centres, aas, omegas,  phis, onsites,  t)
    return -1j*np.dot(H, psi)

def F_TS(t, psi, N, centre, a, omega, phi, onsite):
    H = HT_TS(N, centre, a, omega, phi, onsite, t)
    return -1j*np.dot(H, psi)

# linear moving potential
def F_Linear(t, psi, N, a, omega, phi):
    H = HT_Linear(N, a, omega, t, phi)
    return -1j*np.dot(H, psi)

# no energy offset at all
def F_0(t, psi, N):
    H = H0(N)
    return -1j*np.dot(H, psi)


def F_HF(t, psi, HF):
    return -1j*np.dot(HF, psi)

def F_General(t, psi, N, centre, func, params, circle_boundary):
    H = HT_General(N, centre, func, params, circle_boundary, t)
    return -1j*np.dot(H, psi)

def F_0_T(t, psi, N, centre):
    H = H_0_T(N, centre, t)
    return -1j*np.dot(H, psi)

def F_SSHModel(t, psi, N, aas, omegas, phis, onsites):
    H = HT_SSHModel(N, aas, omegas,  phis, onsites,  t)
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

# import math

# def SigFig(x, digits=6):
#     if x == 0 or not math.isfinite(x):
#         return x
#     digits -= math.ceil(math.log10(np.real(x)))
#     return np.round(x, digits)

def SolveSchrodingerGeneral(N,centre,func,params, tspan, nTimesteps, psi0, circle_boundary = 0):
    
    rtol=1e-11
    # points to calculate the matter wave at
    t_eval = np.linspace(tspan[0], tspan[1], nTimesteps+1, endpoint=True)
    sol = solve_ivp(lambda t,psi: F_General(t, psi, 
                                          N, centre, func, params, circle_boundary), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
    sol=sol.y
    return sol

def SolveSchrodinger(form, rtol, N, centre, a, omega, phi, tspan, nTimesteps, psi0, onsite=0):
    """
    Solve Schrodinger Equation for oscilating Hamiltonian
    Oscillating single site energy, H[centre][centre] = a cos(omega t + phi) (when form = "SS-p")
    tspan = [tstart, tend]
    Initial conditions given by psi0 (initial matter wave)
    
    form = "DS-p" gives double site shaking
    
    form = "linear" will shake whole lattice. This is useful to check known results.
    
    form = "SSDF-p" will shake single site with two frequencies (non time reversal symmetric)
    
    IMPORTANT - nTimesteps is how many steps. This is not how many points we need to calculate the matter wave at;
        we calculate the matter wave at nTimesteps + 1 points. This gives nTimesteps steps. 
    """
    
    # points to calculate the matter wave at
    t_eval = np.linspace(tspan[0], tspan[1], nTimesteps+1, endpoint=True)
        
    if form=="DS-p" or form == "SSDF-p":
        omega1 = omega[0]
        omega2 = omega[1]
        phi1 = phi[0]
        phi2 = phi[1]
        onsite1 = onsite[0]
        onsite2 = onsite[1]
        
    elif form == "StepFunc":
         sol = solve_ivp(lambda t,psi: F_StepFunc(t, psi, 
                                                  N, centre, a, omega, phi, onsite),
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
         sol=sol.y
    
    elif form == "StepFuncGen":
        # NB: centre, a, omega, phi and onsite are lists of at least one object
         sol = solve_ivp(lambda t,psi: F_StepFuncGen(t, psi, 
                                                  N, centre, a, omega, phi, onsite),
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
         sol=sol.y
        
    
    if form == "TS-p":
        sol = solve_ivp(lambda t,psi: F_TS(t, psi, 
                                          N, centre, a, omega, phi, onsite), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
        
    elif form == 'SS-p' or form =="SS-p-RemoveGauge":
        sol = solve_ivp(lambda t,psi: F_SS(t, psi, 
                                           N, centre, a, omega, phi, onsite), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
    
    elif form == "Circle":
        sol = solve_ivp(lambda t,psi: F_Circle(t, psi, 
                                           N, centre, a, omega, phi, onsite), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
    
        
    elif form == 'DS-p':
        sol = solve_ivp(lambda t,psi: F_DS(t, psi,
                                           N, centre, a, omega1, omega2, 
                                           phi1, phi2, onsite1, onsite2), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
    
    elif form == 'SSDF-p':
        sol = solve_ivp(lambda t,psi: F_SSDF(t, psi, 
                                             N, centre, a, omega1, omega2,
                                             phi1, phi2, onsite), 
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
        
    elif form == "H0T":
        sol = solve_ivp(lambda t,psi: F_0_T(t, psi, 
                                N, centre), 
                t_span=tspan, y0=psi0, rtol=rtol, 
                atol=rtol, t_eval=t_eval,
                method='RK45')    
        
    elif form == "SSHModel":
        sol = solve_ivp(lambda t,psi: F_SSHModel(t, psi, N, a, omega, phi, onsite), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
        
        
    elif form == 'numericalG-SS-p':
        #get numerically calculated effective Hamiltonian
        _, HF = CreateHF('SS-p', rtol, N, centre, a, phi, omega)
        #diagonalise
        evals, evecs= GetEvalsAndEvecsGen(HF)
        # get initial state, psi0, written in basis of evecs, find coefficients
        coeffs =  np.dot(np.conj(evecs.T), psi0)
        sol = [np.dot(evecs, coeffs*exp(-1j*evals*t)) for t in t_eval]
        sol = np.vstack(sol).T
     
        
    return sol


def SolveSchrodingerTimeIndependent(hamiltonian, tspan, nTimesteps, psi0):
    t_eval = np.linspace(tspan[0], tspan[1], nTimesteps+1, endpoint=True)
    #diagonalise hamiltonian
    evals, evecs = GetEvalsAndEvecsGen(hamiltonian)
    coeffs =  np.dot(np.conj(evecs.T), psi0)
    sol = [np.dot(evecs, coeffs*exp(-1j*evals*t)) for t in t_eval]
    sol = np.vstack(sol).T
    return sol
    
A = np.array([[1,0],[50,99]])
psi = np.array([1, 1j])

X = np.dot(A, psi)

def CreateHF(form, rtol, N, centre, a, omega, phi, onsite): 

    assert(form in ['linear', "linear-p", "SS-p", "DS-p", "SSDF-p", "StepFunc", "StepFuncGen", "SSHModel", "SS-p-RemoveGauge"])
    T = 2*pi/np.min(omega)
    tspan = (0,T)
    UT = np.zeros([N,N], dtype=np.complex_)
    nTimesteps = 100
    for A_site_start in range(N):
    #    print(A_site_start)
        psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
        sol = SolveSchrodinger(form, rtol, N, centre, a, omega, phi, tspan, nTimesteps, psi0, onsite)
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
    """ This should probs be 5? """
    assert(np.all(0 == (HFr - np.conj(HFr.T))))

    return UT, HF



    
def CreateHFGeneral(num_sites, centre, func, params, T, circle_boundary=0, t0=0, hermitian_accuracy_dp=7): 
    """
    t0 is fraction of T
    """
    tspan = (t0*T,T+t0*T)
    UT = np.zeros([num_sites, num_sites], dtype=np.complex_)
    n_timesteps = 100
    
    for A_site_start in range(num_sites):
        psi0 = np.zeros(num_sites, dtype=np.complex_); psi0[A_site_start] = 1
        sol = SolveSchrodingerGeneral(num_sites, centre, func, params, tspan, n_timesteps, psi0, circle_boundary=circle_boundary)
        UT[:,A_site_start]=sol[:,-1] 
    
    # evals_U, evecs = eig(UT)
    evals_U, evecs = GetEvalsAndEvecsGen(UT) #evals can be imaginary
    evals_H = 1j / T *log(evals_U)
    
    HF = np.zeros([num_sites,num_sites], dtype=np.complex_)
    for i in range(num_sites):
        term = evals_H[i]*np.outer(evecs[:,i], np.conj(evecs[:,i]))
        HF = HF+term

    HF = RoundComplex(HF, hermitian_accuracy_dp)
    # assert(np.all(0 == (HFr - np.conj(HFr.T))))
    return UT, HF
    # if np.all(0 == (HF - np.conj(HF.T))):
    #     return UT, HF
    # else:
    #     return np.nan, np.nan
    


def hoppingHF(N, centre, a, omega, phi):
    HF =  np.zeros([N,N], dtype=np.complex_)
    HF = HF + np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1) 
    entry = -exp(-1j*a*sin(phi)/omega)*jv(0, a/omega)
    HF[centre][centre+1] = entry # the term we collect for hoppping
    HF[centre][centre-1] = entry
    HF[centre+1][centre] = np.conj(entry)
    HF[centre-1][centre] = np.conj(entry)
    HFr = RoundComplex(HF, 7)
    assert(np.all(0 == (HFr - np.conj(HFr.T))))
    return HF, entry


#%%


def formatcomplex(num, dp):
    return ("{:."+str(dp)+"f}").format(num.real) + " + " + ("{:."+str(dp)+"f}").format(num.imag) + "i"

def plotevecs(evecs, N, func, colour, title, ypos=0.934):
    sz = 1
    num = 7
    fig, ax = plt.subplots(nrows = num, ncols = num, sharex=True,
                           sharey=True,
                           figsize=(sz*num,sz*num))

    for i in range(num):
        for j in range(num):
            evec1 = evecs[:,num*i + j]
    
            ax[i,j].plot(range(N), func(evec1), color=colour, linewidth=0.6) 

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
    Make first nonzero element of an evec real and positive
    """
    evecs0_R = RoundComplex(evecs0, 5)
    for vec in range(N):
        #index of first non zero element of this vector
        firstnonzero = (evecs0_R[:,vec]!=0).argmax()
        #make this value real and positive, so arg(this value)=2*pi*n for integer n
        angle = np.angle(evecs0[firstnonzero,vec])
        evecs0[:,vec] = evecs0[:,vec]*exp(-1j*angle)
    return evecs0


def AlignEvecs(evecs0, evecsP, N):
    evecs0_R = RoundComplex(evecs0, 5)
    evecsP_R = RoundComplex(evecsP, 5)
    
    #flip if needed
    for vec in range(N):
        #if one is negative of the other, for rounded evecs, flip one,
        #could both start at zero
        if np.all(evecs0_R[:,vec]==-evecsP_R[:,vec]):
            evecsP[:,vec] = -evecsP[:,vec]
            #redefine rounded evecsP
            evecsP_R[:,vec] = RoundComplex(evecsP[:,vec], 5)
        # else, make them start at the same value
        elif evecs0_R[0,vec] != evecsP_R[0,vec]:
            frac = evecs0[0,vec] / evecsP[0,vec]
            evecsP[:,vec] = evecsP[:,vec] * frac
            #redefine evecsP_R
            evecsP_R[:,vec] = RoundComplex(evecsP[:,vec], 5)
            
    return evecsP

#%%

from fractions import Fraction 

def PhiString(phi):
    fraction = phi/pi
    fraction = Fraction(fraction).limit_denominator(100)
    numerator = fraction.numerator
    denominator = fraction.denominator
    if numerator == 0:
        return "0"
    elif numerator ==1:
        return r"\pi /"+str(denominator)
    else:
        return str(numerator)+r"\pi / "+str(denominator)
      
        
      #%%
      

def RemoveWannierGauge(matrix, c, N):
    phase = np.angle(matrix[c-1,c])
    phase = phase - np.pi #because it should be np.pi (ie negative)
    gaugeMatrix = np.identity(N, dtype=np.complex128)
    gaugeMatrix[c,c] = np.exp(-1j*phase)
    matrix = np.matmul(np.matmul(np.conj(gaugeMatrix), matrix), gaugeMatrix)
    return matrix
        
#%%

def ListRatioLowerTriangle(a1, b1, a2, b2, a3, b3):
    
    if a1 <=1 and b1 <=1:
        if b1<=a1:  # b is smaller than a so b is y and a is x
            return a1, b1
        else:       # a is smaller than b so a is y and b is x
            return b1, a1
    elif a2 <= 1 and b2 <=1:
        if b2 <=a2:
            return a2, b2
        else:
            return b2, a2
    
    elif a3 <=1 and b3 <=1:
        if b3 <=a3:
            return a3,b3
        else:
            return b3,a3