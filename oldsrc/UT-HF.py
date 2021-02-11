# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:47:33 2020

@author: Georgia
"""

"""
Calculate and plot U(T)
"""

from numpy.linalg import eig
from cmath import phase
import matplotlib.colors as col
norm = col.Normalize(vmin=-1, vmax=1) 
from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.special import jv, jn_zeros
import pandas as pd


N = 41; 
a = 25; b=1; c=1;

#diffs = []
#for i in range(10, 100):
#    evals_0,_ = eig(H(i)) 
#    omega = max(evals_0)- min(evals_0)
#    diffs.append(omega)
#plt.plot(list(range(10, 100)), diffs, label='lowest bandwidth')
#plt.xlabel('lattice size N')
#plt.legend()
#plt.show()

centre =20;

evals_0, _ = eig(H_0(N-centre-1))
omega = max(evals_0) - min(evals_0)


#omega=a/float(jn_zeros(1, 2)[-1]) 
#omega=17; 
phi=0; T=2*pi/omega

tspan = (0,T)
UT = np.zeros([N,N], dtype=np.complex_)


for A_site_start in range(N):
#    print(A_site_start)
    psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
    sol = solve_ivp(lambda t,psi: F_OSC(t, psi, N, centre, a, omega, phi), 
                        tspan, psi0, rtol=1e-7, atol=1e-7)
    UT[:,A_site_start]=sol.y[:,-1]



"""
Plot U(T)
"""

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(12,4))
ax[0].matshow(abs(UT), interpolation='none', cmap='PuOr', norm=norm)
ax[1].matshow(np.real(UT), interpolation='none', cmap='PuOr', norm=norm)
ax[2].matshow(np.imag(UT), interpolation='none', cmap='PuOr', norm=norm)
#ax[3].matshow(np.angle(HF), interpolation='none', cmap='PuOr', norm=norm_phase)
ax[0].set_title('abs')
ax[1].set_title('real')
ax[2].set_title('imag')
#ax[3].set_title('phase')
for i in range(3):
    ax[i].tick_params(axis="x", bottom=True, top=True, labelbottom=True, 
      labeltop=False)
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm))
fig.suptitle('U(T) for '+
#             'moving gaussian potential ('+
             'oscillating potential ('+
              'a='+str(a)+
#              ', b='+str(b)+', c='+str(c)+
              ', omega='+str(omega)+
              ')', fontsize=16)
plt.show()



evals_U, evecs = eig(UT)


"""
Plot U(T) eigenvalues
"""

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(12,4))
ax[0].plot(abs(evals_U), '.')
ax[1].plot(np.real(evals_U), '.')
ax[2].plot(np.imag(evals_U), '.')
ax[0].set_title('abs')
ax[1].set_title('real')
ax[2].set_title('imag')
fig.suptitle('eigenvalues of U(T) for '+
             'oscillating potential ('+
              'a='+str(a)+
#              ', b='+str(b)+', c='+str(c)+
              ', omega='+str(omega)+
              ')', fontsize=16)
plt.show()


"""
Plot eigenvectors
"""

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(12,4))
ax[0].matshow(abs(evecs), interpolation='none', cmap='PuOr', norm=norm)
ax[1].matshow(np.real(evecs), interpolation='none', cmap='PuOr', norm=norm)
ax[2].matshow(np.imag(evecs), interpolation='none', cmap='PuOr', norm=norm)
#ax[3].matshow(np.angle(HF), interpolation='none', cmap='PuOr', norm=norm_phase)
ax[0].set_title('abs')
ax[1].set_title('real')
ax[2].set_title('imag')
fig.suptitle('eigenvectors of U(T) (and HF) for '+
             'oscillating potential ('+
              'a='+str(a)+
#              ', b='+str(b)+', c='+str(c)+
              ', omega='+str(omega)+
              ')', fontsize=16)
for i in range(3):
    ax[i].tick_params(axis="x", bottom=True, top=True, labelbottom=True, 
      labeltop=False)
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm))
fig.text(0.47, -0.05, 'eigenvector', ha='center')
plt.show()


"""
Plot HF eigenvalues
"""

evals_H = 1j / T *log(evals_U)

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(12,4))
ax[0].plot(abs(evals_H), '.')
ax[1].plot(np.real(evals_H), '.')
ax[2].plot(np.imag(evals_H), '.')
ax[0].set_title('abs')
ax[1].set_title('real')
ax[2].set_title('imag')
fig.suptitle('eigenvalues of HF for '+
             'oscillating potential ('+
              'a='+str(a)+
#              ', b='+str(b)+', c='+str(c)+
              ', omega='+str(omega)+
              ')', fontsize=16)
plt.show()



"""
Plot HF
"""

HF = np.zeros([N,N], dtype=np.complex_)
for i in range(N):
    term = evals_H[i]*np.outer(evecs[:,i], evecs[:,i])
    HF = HF+term

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(12,4))
ax[0].matshow(abs(HF), interpolation='none', cmap='PuOr', norm=norm)
ax[1].matshow(np.real(HF), interpolation='none', cmap='PuOr', norm=norm)
ax[2].matshow(np.imag(HF), interpolation='none', cmap='PuOr', norm=norm)
#ax[3].matshow(np.angle(HF), interpolation='none', cmap='PuOr', norm=norm_phase)
ax[0].set_title('abs')
ax[1].set_title('real')
ax[2].set_title('imag')
#ax[3].set_title('phase')
for i in range(3):
    ax[i].tick_params(axis="x", bottom=True, top=True, labelbottom=True, 
      labeltop=False)
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm))
fig.suptitle('Floquet Hamiltonian for '+
#             'moving gaussian potential ('+
             'oscillating potential ('+
              'a='+str(a)+
#              ', b='+str(b)+', c='+str(c)+
              ', omega='+str(omega)+
              ')', fontsize=16)
plt.show()

#%%