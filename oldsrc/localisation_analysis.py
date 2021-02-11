# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:08:46 2020

@author: Georgia
"""


from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.special import jv, jn_zeros
import pandas as pd



#%%
'''
Single time evolution plot
'''
N = 51; A_site_start =25;
a = 25; b=1; c=1; 
omega = 4
#omega=a/float(jn_zeros(1, 3)[-1]); 
phi=0; T=2*pi/omega
centre = 25;
tspan = (0,10)
Nt = 100
t_eval = np.linspace(tspan[0], tspan[1], Nt)

psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
sol = solve_ivp(lambda t,psi: F_0(t, psi, N), 
                    tspan, psi0, t_eval=t_eval, rtol=1e-6, atol=1e-6)

    
plt.figure(figsize=(13,7))
plt.matshow(abs(sol.y), interpolation='none', cmap='Greys', fignum=1)#, extent=[0,100,50,0]) 
x_positions = np.arange(0, Nt, T*(Nt/tspan[1]))
x_labels = list(range(len(x_positions)))
plt.xticks(x_positions, x_labels)
plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=False)
plt.xlabel('oscillations')
plt.ylabel('site')
plt.title('psi evolution; single site energy offset cosine modulation with a='+str(a)+
#          ', b='+str(b)+', c='+str(c)+
          ', omega='+str(omega)+
          ', phi='+str(phi))
#plt.title('moving gaus subtract time average; a='+str(a)+', b='+str(b)+
#          ', c='+str(c)+', omega='+str(omega))
plt.colorbar()
plt.show()           

# degree of localisation
print('localisation: '+str(np.sum(abs(sol.y[A_site_start]))/len(sol.t)))


#%%
'''
Multiple frequency storage 
'''

N = 51; A_site_start = 25;
a = 30;
phi=0; T=2*pi/omega

b = 1; c=1; centre = 25;
tspan = (0,10)
Nt = 100
t_eval = np.linspace(tspan[0], tspan[1], Nt)

df1 = pd.DataFrame(columns=['Hamiltonian', 'method', 'rtol', 'atol', 'a', 'b', 'c',
                            'omega', 'phi', 'localisation'])

for i, omega in enumerate(np.linspace(0.1, 2, 100)):
    print(omega)
    
    psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
    sol = solve_ivp(lambda t,psi: F_OSC(t, psi, N, centre, a, omega, phi),
                    tspan, psi0, rtol=1e-7, atol=1e-7, t_eval=t_eval, method='RK45')#'BDF')
    localisation = np.sum(abs(sol.y[A_site_start]))/len(sol.t)
    print(localisation)
    
    
    df1.loc[i] = ['OSC', 'RK45', 1e-6, 1e-6,
           a, 
#           b, c,
           'na', 'na',
           omega, phi, np.sum(abs(sol.y[A_site_start]))/len(sol.t)]

df = pd.read_csv('/Users/Georgia/Code/MBQD/lattice-simulations/analysis.csv', 
                 index_col=False)
df = df.append(df1, ignore_index=True, sort=False)
df.to_csv('/Users/Georgia/Code/MBQD/lattice-simulations/analysis.csv',
          index=False, columns=['Hamiltonian', 'method', 'rtol', 'atol', 
                                'a', 'b', 'c',
                                'omega', 'phi', 'localisation'])

df = df[df['omega']!=0]

#%%

"""
Multiple frequency plot
"""
df = pd.read_csv('/Users/Georgia/Code/MBQD/lattice-simulations/localisation.csv', 
                 index_col=False)

plt.figure(figsize=(10,5))
for a in [10]:
#    for c in [ '1']:
#    b = '1'
#        b = '1'
#        for phi in [0]:
    
            df_plot = df[(df['Hamiltonian']=='OSC')&
                         (df['method']=='RK45')&(df['a']==a)
                         &(df['rtol']==1e-6)&(df['atol']==1e-6)
#                         & (df['b']==b)&(df['c']==c)
#                         & (df['phi']==phi)
                         ]
            df_plot['a0']= df_plot['a']/df_plot['omega']
            df_plot = df_plot.sort_values(by=['omega'])
            
            plt.plot(df_plot['omega'], df_plot['localisation'], 
                     label=''+
                     'localisation'
#                     'a='+str(a)
#                     'b='+str(b)
#                     'c='+str(c)
    #                 +'phi='+str(phi/pi)+'pi'
        #                     +'a='+str(a)
        #                     +' b='+str(b)+' c='+str(c)
                     )
            plt.plot(df_plot['omega'], [jv(0, a/i) for i in df_plot['omega']], 
                     label='J_0(a0)')
            plt.vlines(a/jn_zeros(0,4), -0.4, 1, 
                       colors='0.5', linestyles='dotted')
            plt.vlines(a/jn_zeros(1,4), -0.4, 1, 
                       colors='r', linestyles='dotted')
            plt.hlines(0.22581196211577448, df_plot['omega'].min(), df_plot['omega'].max(),
                       colors='g', linestyles='dotted')
            roundd = lambda t: round(t, 2)
            turningvals = np.array(list(map(roundd, np.append(a/jn_zeros(0, 3), (a/jn_zeros(1, 3))))))
            plt.xticks(turningvals[turningvals>4])

np.append(a/jn_zeros(0, 4), (a/jn_zeros(1, 4)))
plt.xlabel('omega')
#plt.ylabel('localisation')
#plt.xlim([0, 10])
#plt.title('moving gauss subtract time average')
plt.title('Localisation with'+
#          ' moving gaus potential'+
          ' oscillating on site energy'+
          ' ('+
          'a='+str(a)+
#          ', b='+str(b)+
#          ', c='+str(c)+
#          ', phi='+str(phi/pi)+'pi'+
          ')')
plt.legend()
plt.xlim(xmin=3.7)
#plt.xlim([-0.4, 10.4])
plt.show()
      