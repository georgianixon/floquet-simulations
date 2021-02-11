# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:08:46 2020

@author: Georgia
"""


from numpy import exp, sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.special import jv, jn_zeros
import pandas as pd

#%%

"""
Solve schrodinger eq
"""


#moving gaus subtract time 
def F_MGSTA(t, psi, N, centre, a, b, c, omega, phi):
    return -1j*np.dot(HT_MGSTA(N, centre, a, b, c, omega, t, phi),psi)

# one site cosine 
def F_OSC(t, psi, N, centre, a, omega, phi):
    return -1j*np.dot(HT_OSC(N, centre, a, omega, t, phi),psi)

# one site cosine with centre potential = a not zero
def F_OSCpa(t, psi, N, centre, a, omega, phi):
    return -1j*np.dot(HT_OSCpa(N, centre, a, omega, t, phi),psi)

def F_OSCp2a(t, psi, N, centre, a, omega, phi):
    return -1j*np.dot(HT_OSCp2a(N, centre, a, omega, t, phi),psi)


def F_OSCp1p2a(t, psi, N, centre, a, omega, phi):
    return -1j*np.dot(HT_OSCp1p2a(N, centre, a, omega, t, phi),psi)

# linear moving potential
def F_Linear(t, psi, N, a, omega, phi):
    return -1j*np.dot(HT_Linear(N, a, omega, t, phi), psi)

# no energy offset at all
def F_0(t, psi):
    return -1j*np.dot(H(N), psi)

#%%
'''
Single time evolution plot
'''
N = 51; A_site_start = 25;
a = 25; b=1; c=1; omega=10.1; phi=pi/2; T=2*pi/omega
V_centre = 25;
tspan = (0,10)
Nt = 100
t_eval = np.linspace(tspan[0], tspan[1], Nt)

psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
sol = solve_ivp(lambda t,psi: F_OSC(t, psi, N, V_centre, a, omega, phi), 
                    tspan, psi0, t_eval=t_eval, method='BDF')#'BDF')
    
plt.figure(figsize=(10,5))
plt.matshow(abs(sol.y), interpolation='none', cmap='PuOr')#, extent=[0,100,50,0]) 
x_positions = np.arange(0, Nt, T*(Nt/tspan[1]))
x_labels = list(range(len(x_positions)))
plt.xticks(x_positions, x_labels)
plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
plt.xlabel('oscillations')
plt.ylabel('site')
plt.title('psi evolution, oscillating single site potential (avg>0), a='+str(a)+', omega='+str(omega))
plt.colorbar()
plt.show()           

# degree of localisation
print('localisation: '+str(np.sum(abs(sol.y[A_site_start]))/len(sol.t)))



#%%
'''
Multiple frequency storage 
'''

N = 51; A_site_start = 25;
a = 15;
phi=0; T=2*pi/omega

b = 3; c=1; V_centre = 25;
tspan = (0,10)
Nt = 100
t_eval = np.linspace(tspan[0], tspan[1], Nt)

df1 = pd.DataFrame(columns=['Hamiltonian', 'method','a', 'omega', 'localisation'])

for i, omega in enumerate(np.linspace(1, 8, 50)):
    print(omega)
    
    psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
    sol = solve_ivp(lambda t,psi: F_OSCp1p2a(t, psi, N, V_centre, a, omega, phi), 
                    tspan, psi0, t_eval=t_eval, method='BDF')#'BDF')
    df1.loc[i] = ['OSC+1.2a', 'BDF', a, omega, np.sum(abs(sol.y[A_site_start]))/len(sol.t)]

df = pd.read_csv('/Users/Georgia/Code/MBQD/lattice-simulations/localisation.csv', 
                 index_col=False)
df = df.append(df1, ignore_index=True, sort=False)
df.to_csv('/Users/Georgia/Code/MBQD/lattice-simulations/localisation.csv',
          index=False)

#%%

"""
Multiple frequency plot
"""
df = pd.read_csv('/Users/Georgia/Code/MBQD/lattice-simulations/localisation.csv', 
                 index_col=False)

plt.figure(figsize=(10,5))
for a in [15]:
    df_plot = df[(df['Hamiltonian']=='OSC')&(df['method']=='BDF')&(df['a']==a)]
    df_plot['a0']= df_plot['a']/df_plot['omega']
    df_plot = df_plot.sort_values(by=['omega'])
    
    plt.plot(df_plot['omega'], df_plot['localisation'], 
             label='a='+str(a))
    plt.plot(df_plot['omega'], [jv(0, a/i) for i in df_plot['omega']], 
             label='J_0(a0)')
    plt.vlines(a/jn_zeros(0,4), -0.4, 1, colors='0.5', linestyles='dotted')
plt.xlabel('omega')
plt.ylabel('localisation')
plt.xlim([0, 20])
plt.title('one site cosine modulated hamiltonian (avg>0)')
plt.legend()
plt.show()
      

# degree of localisation
print('localisation: '+str(np.sum(abs(sol.y[A_site_start]))/len(sol.t)))


#%%

fig, ax = plt.subplots(constrained_layout=True)
ax.matshow(abs(sol.y), interpolation='none', extent=[0,100,50,0]) 

plt.plot(x, [jv(0, i) for i in x])
plt.plot(x, [jv(1, i) for i in x])