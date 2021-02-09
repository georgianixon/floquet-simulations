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
                            
#HT_MGSTA(5, 3, 10, 1, 1, 3.3, 0, 0)


"""
Time dependent linear energy offset
"""

def HT_Linear(N, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)  
    for i in range(N):
        matrix[i][i] = a*i*cos(omega*t + phi)
    return matrix

def line(N, a, i):
    return a*(2*i / (N-1) - 1)

#HT_Linear(6, 4, 1, 0, 0)


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
                        
H_varied_hopping(10, 5, [-0.9, -0.7, -0.1])


"""
Create one site cosine modulated energy offset hamiltonian
Centre indexed from 0
"""
def HT_OSC(N, centre, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega*t + phi)
    return matrix

def HT_OSC2(N, centre1, centre2, a1, a2, omega, t, phi1, phi2):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre1][centre1] = a1*cos(omega*t + phi1)
    matrix[centre2][centre2] = a2*cos(omega*t + phi2)
    return matrix

HT_OSC(10, 4, 10, 1, pi, 0)

def HT_OSCpa(N, centre, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega*t + phi)+a
    return matrix

def HT_OSCp2a(N, centre, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega*t + phi)+2*a
    return matrix

def HT_OSCp1p2a(N, centre, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega*t + phi)+1.2*a
    return matrix

#HT_OSCpa(10, 4, 10, 1, pi, 0)

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
    return -1j*np.dot(HT_MG(N, centre, a, b, c, omega, t, phi),psi)
    
#moving gaus subtract time 
def F_MGSTA(t, psi, N, centre, a, b, c, omega, phi):
    return -1j*np.dot(HT_MGSTA(N, centre, a, b, c, omega, t, phi),psi)

# one site cosine 
def F_OSC(t, psi, N, centre, a, omega, phi):
    return -1j*np.dot(HT_OSC(N, centre, a, omega, t, phi),psi)

# one site cosine 
def F_OSC_i(t, psi, N, centre, a, omega, phi):
    return -1*np.dot(HT_OSC(N, centre, a, omega, t, phi),psi)

# two site cosine 
def F_OSC2(t, psi, N, centre1, centre2, a1, a2, omega, phi1, phi2):
    return -1j*np.dot(HT_OSC2(N, centre1, centre2, a1, a2, omega,
                              t, phi1, phi2),psi)

# one site cosine with centre potential = a not zero
def F_OSCpa(t, psi, N, centre, a, b, c, omega, phi):
    return -1j*np.dot(HT_OSCpa(N, centre, a, omega, t, phi),psi)

def F_OSCp2a(t, psi, N, centre, a, b, c, omega, phi):
    return -1j*np.dot(HT_OSCp2a(N, centre, a, omega, t, phi),psi)


def F_OSCp1p2a(t, psi, N, centre, a, b, c, omega, phi):
    return -1j*np.dot(HT_OSCp1p2a(N, centre, a, omega, t, phi),psi)

# linear moving potential
def F_Linear(t, psi, N, a, omega, phi):
    return -1j*np.dot(HT_Linear(N, a, omega, t, phi), psi)

# no energy offset at all
def F_0(t, psi, N):
    return -1j*np.dot(H_0(N), psi)


def F_HF(t, psi, HF):
    return -1j*np.dot(HF, psi)


#%%
'''
Create HF OFC
'''


# import numpy as np
# from sympy import Matrix, init_printing
# init_printing()

# display(Matrix(HF))


#%%



from numpy.linalg import eig

def solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi, tspan, psi0, avg=False):
    """
    solve time dependent schrodinger eq given initial conditions psi0, over
    time tspan, for Hamiltonian signified by 'form'
    """
    
    t_eval = np.linspace(tspan[0], tspan[1], 100)
    
    if avg == True:
        UT, HF = create_HF(form, rtol, N, centre, a, b, c,  phi, omega)
        # find eigenvalues
        
        sol= solve_ivp(lambda t,psi: F_HF(t, psi, HF), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
    
    
    elif form=='OSC' or form=='OSC_sort':
        sol= solve_ivp(lambda t,psi: F_OSC(t, psi, 
                           N, centre,
                             a,
                             omega, phi), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
    elif form=='OSC_i':
        sol= solve_ivp(lambda t,psi: F_OSC_i(t, psi, 
                           N, centre,
                             a,
                             omega, phi), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
    elif form=='MG':
        sol = solve_ivp(lambda t,psi: F_MG(t, psi, 
                           N, centre,
                             a,
                             b, c,
                             omega, phi), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
        
    elif form =='linear':
        sol= solve_ivp(lambda t,psi: F_Linear(t, psi, N, a, omega, phi), 
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
    
        
    elif form == 'MGSTA':
        sol = solve_ivp(lambda t,psi: F_MGSTA(t, psi, N, 
                                         centre,
                                         a, 
                                         b,
                                         c,
                                         omega, 
                                         phi), 
                        tspan, psi0, rtol=rtol, atol=rtol)
        
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
        
        return 0, HF
         
        
    else:
    
        T=2*pi/omega
        tspan = (0,T)
        UT = np.zeros([N,N], dtype=np.complex_)
        
        for A_site_start in range(N):
        #    print(A_site_start)
            psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
            sol = solve_schrodinger(form, rtol, N, centre, a, b, c, omega, phi,  tspan, psi0)
            UT[:,A_site_start]=sol.y[:,-1]
        
        # print(time.time()-start, 'seconds.')
        
        evals_U, evecs = eig(UT)
        
        if form=='OSC_sort':
            '''new order sort thing'''
            idx = evals_U.argsort()[::-1]   
            evals_U = evals_U[idx]
            evecs = evecs[:,idx]
    
        evals_H = 1j / T *log(evals_U)
        
        HF = np.zeros([N,N], dtype=np.complex_)
        for i in range(N):
            term = evals_H[i]*np.outer(evecs[:,i], evecs[:,i])
            HF = HF+term
            
        # print('   ',time.time()-start, 's')
        
        return UT, HF






