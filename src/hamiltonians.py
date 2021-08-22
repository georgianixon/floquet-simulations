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

def GetEvalsAndEvecs(HF):
    """
    Get e-vals and e-vecs of Hamiltonian HF.
    Order Evals and correspoinding evecs by smallest eval first.
    Set the gauge for each evec; choosing the first non-zero element to be real and positive.
    Note that the gauge may be changed later by multiplying any vec arbitrarily by a phase. 
    """
    #order by evals, also order corresponding evecs
    evals, evecs = eig(HF)
    idx = np.real(evals).argsort()
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    #make first element of evecs real and positive
    for vec in range(np.size(HF[0])):
        
        # Find first element of the first eigenvector that is not zero
        firstNonZero = (evecs[:,vec]!=0).argmax()
        #find the conjugate phase of this element
        conjugatePhase = np.conj(evecs[firstNonZero,vec])/np.abs(evecs[firstNonZero,vec])
        #multiply all elements by the conjugate phase
        evecs[:,vec] = conjugatePhase*evecs[:,vec]

    # check that the evals are real
    if np.all((np.round(np.imag(evals),7) == 0)) == True:
        return np.real(evals), evecs
    else:
        print('evals are imaginary!')
        return evals, evecs

    
    
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

"""
No energy offset
"""
def H_0(N):
    return np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)      



def HT_General(N, centres, funcs, paramss, t):
    H = H_0(N)
    
    numOfShakes = len(centres)
    if numOfShakes == 1:
        H[centres,centres] = funcs(paramss, t)
    else:    
        for i in range(numOfShakes):
            func = funcs[i]
            H[centres[i],centres[i]] = func(paramss[i], t)
    return H


"""
Functions to Solve Schrodinger eq
"""


# one site cosine 
def F_SS(t, psi, N, centre, a, omega, phi, onsite):
    return -1j*np.dot(HT_SS(N, centre, a, omega, phi, onsite, t), psi)

def F_DS(t, psi, N, centre, a, omega1, omega2, phi1, phi2, onsite1, onsite2):
    return -1j*np.dot(HT_DS(N, centre, a, omega1, omega2, phi1, phi2, onsite1, onsite2, t), psi)

def F_SSDF(t, psi, N, centre, a, omega1, omega2, phi1, phi2, onsite):
    return -1j*np.dot(HT_SSDF(N, centre, a, omega1, omega2, phi1, phi2, onsite, t), psi)


def F_TS(t, psi, N, centre, a, omega, phi, onsite):
    return -1j*np.dot(HT_TS(N, centre, a, omega, phi, onsite, t), psi)

# linear moving potential
def F_Linear(t, psi, N, a, omega, phi):
    return -1j*np.dot(HT_Linear(N, a, omega, t, phi), psi)

# no energy offset at all
def F_0(t, psi, N):
    return -1j*np.dot(H_0(N), psi)


def F_HF(t, psi, HF):
    return -1j*np.dot(HF, psi)

def F_General(t, psi, N, centre, func, params):
    H = HT_General(N, centre, func, params, t)
    return -1j*np.dot(H, psi)


def ConvertComplex(s):
    """
    For retrieving complex numbers from csv's
    """
    return np.complex128(s.replace('i', 'j').replace('*I', 'j').replace('*^', 'e'))

def RoundComplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j

def SolveSchrodingerGeneral(N,centre,func,params, tspan, nTimesteps, psi0,):
        
    
    rtol=1e-11
    # points to calculate the matter wave at
    t_eval = np.linspace(tspan[0], tspan[1], nTimesteps+1, endpoint=True)
    sol = solve_ivp(lambda t,psi: F_General(t, psi, 
                                          N, centre, func, params), 
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
    
        
    if form=="DS-p" or form == "SSDF-p":
        omega1 = omega[0]
        omega2 = omega[1]
        phi1 = phi[0]
        phi2 = phi[1]
        onsite1 = onsite[0]
        onsite2 = onsite[1]
        
    # points to calculate the matter wave at
    t_eval = np.linspace(tspan[0], tspan[1], nTimesteps+1, endpoint=True)
    
    if form == "TS-p":
        sol = solve_ivp(lambda t,psi: F_TS(t, psi, 
                                          N, centre, a, omega, phi, onsite), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
        
    elif form == 'SS-p':
        sol = solve_ivp(lambda t,psi: F_SS(t, psi, 
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
        
        
    elif form == 'numericalG-SS-p':
        #get numerically calculated effective Hamiltonian
        _, HF = CreateHF('SS-p', rtol, N, centre, a, phi, omega)
        #diagonalise
        evals, evecs= GetEvalsAndEvecs(HF)
        # get initial state, psi0, written in basis of evecs, find coefficients
        coeffs =  np.dot(np.conj(evecs.T), psi0)
        sol = [np.dot(evecs, coeffs*exp(-1j*evals*t)) for t in t_eval]
        sol = np.vstack(sol).T
        
        
    return sol




def CreateHF(form, rtol, N, centre, a,phi, omega): 

    assert(form in ['linear', "linear-p", "SS-p", "DS-p", "SSDF-p"])
    if form == "DS-p" or form =="SSDF-p":
        T = 2*pi/omega[0]
    else:
        T=2*pi/omega
    tspan = (0,T)
    UT = np.zeros([N,N], dtype=np.complex_)
    
    for A_site_start in range(N):
    #    print(A_site_start)
        psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
        sol = SolveSchrodinger(form, rtol, N, centre, a, omega, phi, tspan, 100, psi0)
        UT[:,A_site_start]=sol[:,-1] 
    
    # print(time.time()-start, 'seconds.')
    
    evals_U, evecs = eig(UT)
    evals_H = 1j / T *log(evals_U)
    
    HF = np.zeros([N,N], dtype=np.complex_)
    for i in range(N):
        term = evals_H[i]*np.outer(evecs[:,i], np.conj(evecs[:,i]))
        HF = HF+term
        
    # print('   ',time.time()-start, 's')
    HFr = RoundComplex(HF, 5)
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
    HFr = RoundComplex(HF, 7)
    assert(np.all(0 == (HFr - np.conj(HFr.T))))
    return HF, entry


#%%


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
        str(numerator)+r"\pi / "+str(denominator)
        
        