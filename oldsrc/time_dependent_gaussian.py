# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:08:46 2020

@author: Georgia
"""


from numpy import exp, sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt

"""
Create gaussian profile
"""

def moving_gaus(a, b, c, x, omega, t, phi):
    return a*exp(-(x - b*sin(omega*t +phi))**2/(2*c**2))

moving_gaus(10, 1, 1, 0, 3.3, 0, 0)

x = np.arange(0, 10, 0.1)

a = 10; b = 1; c = 1; omega = 3.3; t =0; phi=0;
plt.plot(x, [moving_gaus(a, b, c, i, omega, t, phi) for i in x])
plt.xlabel('x')
plt.show()



t = np.arange(0, pi, 0.1)

a = 1; b = 1; c = 1; omega = 1; x = 0; phi = 0;
plt.plot(t, [moving_gaus(a, b, c, x, omega, i, phi) for i in t])
plt.plot(t, 0.2*cos(2*t)+0.8)
plt.xlabel('t')
plt.show()

#%%
"""
Create time average gaussian profile
"""
from scipy.integrate import quad


function_NIntegrate = lambda t, b, c, x: exp(x*b*sin(t)/c**2)*exp(b**2*cos(2*t)/4/c**2)
def time_average(a, b, c, x):
    res, err = quad(function_NIntegrate, 0, 2*pi, args=(b, c, x))
    return a/2/pi*exp((-2*x**2 - b**2)/4/c**2)*res

a = 10; b = 1; c = 1; omega = 3.3; t =0; phi=0;
x = np.arange(-5, 5, 0.1)
y = [time_average(a, b, c, i) for i in x]
plt.plot(x, y)   
plt.show()

#%%
"""
Create moving gaussian subtract time average
"""
def moving_gaus_subtract_time_average(a, b, c, x, omega, t, phi):
    return moving_gaus(a, b, c, x, omega, t, phi) - time_average(a, b, c, x)

a = 10; b = 1; c = 1; omega = 3.3; t =0; phi=0;
x = np.arange(-5, 5, 0.1)
plt.plot(x, [moving_gaus_subtract_time_average(a, b, c, i, omega, t, phi) for i in x])
plt.show()

#%%
"""
Create time-dependent hamiltonian where the energy offset is moving gaus - time
average
"""

#remember centre is indexed from 0!!

def HT_MGSTA(N, centre, a, b, c, omega, t, phi):
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                matrix[i][j] = moving_gaus_subtract_time_average(a, b, c, i - centre, omega, t, phi)
            elif abs(i - j) == 1:
                matrix[i][j] = -1
    return matrix
                            
G = HT_MGSTA(6, 3, 10, 1, 1, 3.3, 0, 0)


#%%
#remember centre is indexed from 0!!
"""
Create time independent hopping with modulated hoppings
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



#%%
#remember centre is indexed from 0!!
"""
Create one site cosine modulated energy offset hamiltonian
"""
def HT_OSCM(N, centre, a, omega, t, phi):
    matrix = np.diag(-np.ones(N-1),-1)+np.diag(-np.ones(N-1),1)          
    matrix[centre][centre] = a*cos(omega*t + phi)
    return matrix

HT_OSCM(10, 4, 10, 1, pi, 0)

#%%

"""
Solve schrodinger eq
"""
from scipy.integrate import solve_ivp

def F_MGTSA(t, psi):
    N = 81; centre=40; a=10; b=1; c=1; omega=3.3; phi=0;
    psi1 = -1j*np.dot(HT_MGSTA(N, centre, a, b, c, omega, t, phi),psi)
    return psi1

def F_OSCM(t, psi, N, centre, a, omega, phi):
    psi1 = -1j*np.dot(HT_OSCM(N, centre, a, omega, t, phi),psi)
    return psi1

N = 50; A_site_start = 25;
V_centre = 25; a=10; omega=4.2; phi=pi/2;
tspan = (0,10)
t_eval = np.linspace(tspan[0], tspan[1], 100)
psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;

sol = solve_ivp(lambda t,psi: F_OSCM(t, psi, N, V_centre, a, omega, phi), 
                tspan, psi0, t_eval=t_eval)

plt.matshow(abs(sol.y)) 
plt.show()     
#plt.matshow(np.real(sol.y)) 
#plt.show()     
#plt.matshow(np.imag(sol.y))
#plt.show()        

# degree of localisation
print('localisation: '+str(np.sum(abs(sol.y[A_site_start]))/len(sol.t)))
