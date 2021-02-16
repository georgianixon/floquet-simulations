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

"""
Create gaussian profile
"""

def moving_gaus(a, b, c, x, omega, t, phi):
    return a*exp(-(x - b*sin(omega*t +phi))**2/(2*c**2))

#moving_gaus(10, 1, 1, 0, 3.3, 0, 0)
#
#x = np.arange(0, 10, 0.1)
#
#a = 25; b = 7; c = 4; omega = 1; t =0; phi=0;
#plt.plot(x, [moving_gaus(a, b, c, i, omega, t, phi) for i in x])
#plt.xlabel('x')
#plt.title('gaus')
#plt.show()
#
#t = np.arange(0, pi, 0.1)
#
#a = 25; b = 1; c = 4; omega = 1; x = 0; phi = 0;
#plt.plot(t, [moving_gaus(a, b, c, x, omega, i, phi) for i in t], label='energy offset')
#plt.plot(t, 0.4*cos(2*t)+24.6, label='cos function')
#plt.title('energy offset at central site (a='+str(a)+
#                                ', b='+str(b)+', c='+str(c)+')')
#plt.xlabel('time')
#plt.legend()
#plt.show()
    

#%%
"""
Create moving Gaus Hamiltonian
"""
def HT_MG(N, centre, a, b, c, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)  
    for i in range(N):
        matrix[i][i] = moving_gaus(a, b, c, i - centre, omega, t, phi)
    return matrix
                     
#HT_MG(5, 3, 5, 1, 1, 1, 0,0)

"""
Create time average gaussian profile
"""
from scipy.integrate import quad

function_NIntegrate = lambda t, b, c, x: exp(x*b*sin(t)/c**2)*exp(b**2*cos(2*t)/4/c**2)
def time_average(a, b, c, x):
    res, err = quad(function_NIntegrate, 0, 2*pi, args=(b, c, x))
    return a/2/pi*exp((-2*x**2 - b**2)/4/c**2)*res

#a = 30; b = 1; c = 1; omega = 3.3; t =0; phi=0;
#x = np.arange(-5, 5, 0.1)
#y = [time_average(a, b, c, i) for i in x]
#plt.plot(x, y)   
#plt.show()

"""
Create moving gaussian subtract time average
"""
def moving_gaus_subtract_time_average(a, b, c, x, omega, t, phi):
    return moving_gaus(a, b, c, x, omega, t, phi) - time_average(a, b, c, x)

#a = 10; b = 1; c = 1; omega = 3.3; t =0; phi=0;
#x = np.arange(-5, 5, 0.1)
#plt.plot(x, [moving_gaus_subtract_time_average(a, b, c, i, omega, t, phi) for i in x])
#plt.show()


"""
Create time-dependent hamiltonian where the energy offset is moving gaus - time
average
Centre indexed from 0
"""
def HT_MGSTA(N, centre, a, b, c, omega, t, phi):
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                matrix[i][j] = moving_gaus_subtract_time_average(a, b, c, i - centre, omega, t, phi)
            elif abs(i - j) == 1:
                matrix[i][j] = -1
    return matrix


"""
Time dependent linear energy offset
"""

def HT_Linear(N, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)  
    for i in range(N):
        matrix[i][i] = a*i*cos(omega*t + phi)
    return matrix

# def line(N, a, i):
#     return a*(2*i / (N-1) - 1)


"""
Create time independent hopping with modulated hoppings
Centre indexed from 0
"""
def H_varied_hopping(N, centre, alterations):
    radius = len(alterations)
    alterations = alterations+ list(reversed(alterations))
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)  
    for i in range(N):
        for j in range(N):
            if abs(i-j) ==1:
                if min(i,j) > centre - radius - 1 and min(i,j) < centre + radius:
                    if i < j: # top row
                        matrix[i][j] = alterations[i- (centre - radius)]
                    else: # bottom row
                        matrix[i][j] = alterations[j- (centre - radius)]
    return matrix
                        



"""
Create one site cosine modulated energy offset hamiltonian
Centre indexed from 0
"""
def HT_OSC(N, centre, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega*t + phi)
    return matrix



"""
No energy offset
"""
def H_0(N):
    return np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          

#%%

"""
Functions to Solve Schrodinger eq
"""


# moving gaus
def F_MG(t, psi, N, centre, a, b, c, omega, phi):
    return 1j*np.dot(HT_MG(N, centre, a, b, c, omega, t, phi),psi)
    
#moving gaus subtract time 
def F_MGSTA(t, psi, N, centre, a, b, c, omega, phi):
    return 1j*np.dot(HT_MGSTA(N, centre, a, b, c, omega, t, phi),psi)

# one site cosine 
def F_OSC(t, psi, N, centre, a, omega, phi):
    return 1j*np.dot(HT_OSC(N, centre, a, omega, t, phi),psi)

# linear moving potential
def F_Linear(t, psi, N, a, omega, phi):
    return 1j*np.dot(HT_Linear(N, a, omega, t, phi), psi)

# no energy offset at all
def F_0(t, psi, N):
    return 1j*np.dot(H_0(N), psi)


def F_HF(t, psi, HF):
    return 1j*np.dot(HF, psi)



#%%



from scipy.linalg import eig as eig
from scipy.linalg import expm

def solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi, tspan, n_timesteps, psi0):
    """
    solve time dependent schrodinger eq given initial conditions psi0, over
    time tspan, for Hamiltonian signified by 'form'
    """
    
    t_eval = np.linspace(tspan[0], tspan[1], n_timesteps+1)
    
    if form=='OSC':
        sol= solve_ivp(lambda t,psi: F_OSC(t, psi, 
                           N, centre,
                             a,
                             omega, phi), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
        

    elif form=='MG':
        sol = solve_ivp(lambda t,psi: F_MG(t, psi, 
                           N, centre,
                             a,
                             b, c,
                             omega, phi), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
        
    elif form =='linear':
        sol= solve_ivp(lambda t,psi: F_Linear(t, psi, N, a, omega, phi), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        sol=sol.y
        
    elif form == 'MGSTA':
        sol = solve_ivp(lambda t,psi: F_MGSTA(t, psi, N, 
                                         centre,
                                         a, 
                                         b,
                                         c,
                                         omega, 
                                         phi), 
                        tspan, psi0, rtol=rtol, atol=rtol)
        sol=sol.y
    
    elif form== 'theoretical_hermitian':
        _, HF = create_HF(form, None, N, centre, a, None, None, phi, omega)
        
        #find solution at different times
        sol = [np.dot(expm(-1j*HF*T), psi0) for T in t_eval]
        #turn vector into same form as the solvers have
        sol = np.vstack(sol).T
        
        
        
        # assert(np.all(0 == (HF - np.conj(HF.T))))
        
        # evals, evecs= eig(HF)
        # coeffs =  np.dot(np.conj(evecs.T), psi0)
        # psi0_n =np.dot(evecs, coeffs) # check = psi0?
        
        # sol = [np.dot(evecs, coeffs*exp(-1j*evals*t)) for t in t_eval]
        # sol = np.vstack(sol).T
        
    elif form == 'theoretical':
        _, HF = create_HF(form, None, N, centre, a, None, None, phi, omega)
        #find solution at different times
        sol = [np.dot(expm(-1j*HF*T), psi0) for T in t_eval]
        #turn vector into same form as the solvers have
        sol = np.vstack(sol).T
        
    return sol


from scipy.special import jv
from numpy import exp, sin

def create_HF(form, rtol, N, centre, a,b, c,phi, omega): 
    
    if form == 'theoretical':
        HF =  np.zeros([N,N], dtype=np.complex_)
        HF = HF + np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1) 
        entry = exp(1j*a*sin(phi)/omega)*jv(0, a/omega) 
        HF[centre][centre+1] = entry
        HF[centre][centre-1] = entry
        HF[centre+1][centre] = entry
        HF[centre-1][centre] = entry
        
        return None, HF
    
    if form == 'theoretical_hermitian':
        HF =  np.zeros([N,N], dtype=np.complex_)
        HF = HF + np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1) 
        entry = exp(1j*a*sin(phi)/omega)*jv(0, a/omega) 
        HF[centre][centre+1] = entry
        HF[centre][centre-1] = entry
        HF[centre+1][centre] = np.conj(entry)
        HF[centre-1][centre] = np.conj(entry)
        
        return None, HF     
        
    else:
    
        T=2*pi/omega
        tspan = (0,T)
        UT = np.zeros([N,N], dtype=np.complex_)
        
        for A_site_start in range(N):
        #    print(A_site_start)
            psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
            sol = solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi, tspan, 100, psi0)
            UT[:,A_site_start]=sol.y[:,-1]
        
        # print(time.time()-start, 'seconds.')
        
        evals_U, evecs = eig(UT)
    
        evals_H = 1j / T *log(evals_U)
        
        HF = np.zeros([N,N], dtype=np.complex_)
        for i in range(N):
            term = evals_H[i]*np.outer(evecs[:,i], evecs[:,i])
            HF = HF+term
            
        # print('   ',time.time()-start, 's')
        
        return UT, HF






