# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:01:15 2020

@author: Georgia
"""

from numpy.linalg import eig
from cmath import phase
import matplotlib.colors as col

from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.special import jv, jn_zeros
import pandas as pd
import time

def bandwidth(N):
    evals, _ = eig(H_0(N))
    return max(evals) - min(evals)

#%%

"""
For single site oscillation
"""
norm = col.Normalize(vmin=-1, vmax=1) 
N = 31; 
centre=15;
a=30;
phi=0; 
omega = 8
T=2*pi/omega

tspan = (0,T)
UT = np.zeros([N,N], dtype=np.complex_)
start = time.time()
for A_site_start in range(N):
#    print(A_site_start)
    psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
    sol = solve_ivp(lambda t,psi: F_OSC(t, psi, N, 
                                         centre,
                                         a, 
                                         omega, 
                                         phi), 
                        tspan, psi0, rtol=1e-7, atol=1e-7)
    UT[:,A_site_start]=sol.y[:,-1]

print(time.time()-start, 'seconds.')
"""
Plot HF
"""
evals_U, evecs = eig(UT)
evals_H = 1j / T *log(evals_U)

HF = np.zeros([N,N], dtype=np.complex_)
for i in range(N):
    term = evals_H[i]*np.outer(evecs[:,i], evecs[:,i])
    HF = HF+term
    
sz = 3
fig, ax = plt.subplots(figsize=(sz,sz))
ax.matshow(np.real(HF), interpolation='none', cmap='PuOr', norm=norm)
#ax.set_title('real')
ax.tick_params(axis="x", bottom=True, top=False,  labelbottom=True, 
  labeltop=False)
ax.set_xlabel('m')
ax.set_ylabel('n', rotation=0, labelpad=10)

cax = plt.axes([1, 0.05, 0.06, 0.9])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)

#fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm))
#fig.suptitle('FH, '+
#             'oscillating V ('+
#             'N='+str(N)+
#             ', o='+str(centre)+
#              ', a='+str(a)+
#              ', omega='+str(("{:.2f}".format(omega)))+
#              ', phi='+str("{:.1f}".format(phi/pi))+
#              'pi'
#              ')', fontsize=16)
#             
fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
        'first_year_report/HF,F=30,w=8,ph=0.pdf', 
        format='pdf', bbox_inches='tight')
plt.show()



#%%
"""
HF Plot for Moving Gaus
"""

norm = col.Normalize(vmin=-1, vmax=1) 
N = 31; 
centre=15;
a=30;
b = 1;
c = 1;
phi=0; 
omega = 5

T=2*pi/omega

tspan = (0,T)
UT = np.zeros([N,N], dtype=np.complex_)
start = time.time()
for A_site_start in range(N):
#    print(A_site_start)
    psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
    sol = solve_ivp(lambda t,psi: F_MGSTA(t, psi, N, 
                                         centre,
                                         a, 
                                         b,
                                         c,
                                         omega, 
                                         phi), 
                        tspan, psi0, rtol=1e-7, atol=1e-7)
    UT[:,A_site_start]=sol.y[:,-1]

print(time.time()-start, 'seconds.')
"""
Plot HF
"""      
        
      
evals_U, evecs = eig(UT)
evals_H = 1j / T *log(evals_U)

HF = np.zeros([N,N], dtype=np.complex_)
for i in range(N):
    term = evals_H[i]*np.outer(evecs[:,i], evecs[:,i])
    HF = HF+term
    


sz = 6.2
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))
ax[0].matshow(np.abs(HF), interpolation='none', cmap='PuOr',  norm=norm)
ax[1].matshow(np.real(HF), interpolation='none', cmap='PuOr',  norm=norm)
ax[2].matshow(np.imag(HF), interpolation='none', cmap='PuOr',  norm=norm)
#ax[3].matshow(np.angle(HF), interpolation='none', cmap='PuOr', norm=norm_phase)
ax[0].set_title(r'(a)  $\mathrm{Abs}\{G_{n,m}\}$')
ax[1].set_title(r'(b)  $\mathrm{Re}\{G_{n,m}\}$')
ax[2].set_title(r'(c)  $\mathrm{Imag}\{G_{n,m}\}$')

ax[0].set_ylabel('n', rotation=0, labelpad=10)
for i in range(3):
    ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[i].set_xlabel('m')
    
cax = plt.axes([1.03, 0.1, 0.03, 0.8])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
#fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm))
#fig.suptitle('FH, '+
##             'Linear oscillating V ('+
#             'moving gaussian potential ('+
##             'oscillating V ('+
#             'N='+str(N)+
#             ', o='+str(centre)+
##             ', o2='+str(centre2)+
#              ', a='+str(a)+
##              ', a2='+str(a2)+
#              ', b='+str(b)+', c='+str(c)+
#              ', omega='+str(("{:.2f}".format(omega)))+
#              ', phi='+str("{:.1f}".format(phi/pi))+
#              'pi'
##              +', phi2='+str("{:.2f}".format(phi2/pi))+'pi'+
#              ')', fontsize=16)
#             
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#        'first_year_report/HF,MGSTA,a=30,b=1,c=1,w=5,ph=0.pdf', 
#        format='pdf', bbox_inches='tight')
plt.show()
#%%


"""
Specific config we want for the paper
"""

N = 31; 
centre=15;
b = 1;
c = 1; 
omega = 5
T=2*pi/omega
tspan = (0,T)

sz = 7
fig, ax = plt.subplots(nrows=2, ncols=3, 
                       figsize=(sz,sz/2), constrained_layout=True)

for num0, a in enumerate([5]):
    for num1, phi in enumerate([0, pi/2]):

        UT = np.zeros([N,N], dtype=np.complex_)
        for A_site_start in range(N):
        #    print(A_site_start)
            psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
            sol = solve_ivp(lambda t,psi: F_MG(t, psi, N, 
                                                 centre,
                                                 a, 
                                                 b,
                                                 c,
                                                 omega, 
                                                 phi), 
                                tspan, psi0, rtol=1e-7, atol=1e-7)
            UT[:,A_site_start]=sol.y[:,-1]

        """
        Plot HF
        """      
        
      
        evals_U, evecs = eig(UT)
        evals_H = 1j / T *log(evals_U)
        
        HF = np.zeros([N,N], dtype=np.complex_)
        for i in range(N):
            term = evals_H[i]*np.outer(evecs[:,i], evecs[:,i])
            HF = HF+term


        ax[num0*2+num1,0].matshow(np.abs(HF), interpolation='none', cmap='PuOr',  norm=norm)
        ax[num0*2+num1,1].matshow(np.real(HF), interpolation='none', cmap='PuOr',  norm=norm)
        ax[num0*2+num1,2].matshow(np.imag(HF), interpolation='none', cmap='PuOr',  norm=norm)
        #ax[3].matshow(np.angle(HF), interpolation='none', cmap='PuOr', norm=norm_phase)
        ax[num0*2+num1,0].set_title(r'(a)  $\mathrm{Abs}\{G_{n,m}\}$')
        ax[num0*2+num1,1].set_title(r'(b)  $\mathrm{Re}\{G_{n,m}\}$')
        ax[num0*2+num1,2].set_title(r'(c)  $\mathrm{Imag}\{G_{n,m}\}$')

for i in range(4):
    for j in range(3):
        ax[i,j].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
for i in range(4):
    ax[i, 0].set_ylabel('n')
for i in range(3):
    ax[3, i].set_xlabel('m')
    
     

#fig.suptitle('FH, '+
##             'Linear oscillating V ('+
#             'moving gaussian potential ('+
##             'oscillating V ('+
#             'N='+str(N)+
#             ', o='+str(centre)+
##             ', o2='+str(centre2)+
#              ', a='+str(a)+
##              ', a2='+str(a2)+
#              ', b='+str(b)+', c='+str(c)+
#              ', omega='+str(("{:.2f}".format(omega)))+
#              ', phi='+str("{:.1f}".format(phi/pi))+
#              'pi'
##              +', phi2='+str("{:.2f}".format(phi2/pi))+'pi'+
#              ')', fontsize=16)
#             

#plt.subplots_adjust(left=0.125, bottom=0.2, right=0.4, top=0.8, wspace=0.0001, hspace=1)
   
#left  = 0.125  # the left side of the subplots of the figure
#      right = 0.9    # the right side of the subplots of the figure
#      bottom = 0.1   # the bottom of the subplots of the figure
#      top = 0.9      # the top of the subplots of the figure
#      wspace = 0.2   # the amount of width reserved for blank space between subplots
#      hspace = 0.2   # the amount of height reserved for white space between subplots


#cax = plt.axes([0.41, 0.2, 0.008, 0.6])
#fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
#plt.tight_layout()
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#        'first_year_report/HF,MG,a=15,b=1,c=1,various_, various_phi.pdf', 
#        format='pdf', bbox_inches='tight')

plt.show()


#%%
"""
New Gaus... - WE USED THIS ONE
"""


norm = col.Normalize(vmin=-1, vmax=1) 
N = 31; 
centre=15;
a=30;
b = 0.1;
c = 1;
phi=0; 



sz = 6
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(sz,sz/1.42), constrained_layout=True)
    

titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for num0, c in enumerate([0.1, 1]):
    for num1, omega in enumerate([5, 8, 12]):
    
        T=2*pi/omega
        tspan = (0,T)
            
        UT = np.zeros([N,N], dtype=np.complex_)
        start = time.time()
        for A_site_start in range(N):
        #    print(A_site_start)
            psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
            sol = solve_ivp(lambda t,psi: F_MG(t, psi, N, 
                                                 centre,
                                                 a, 
                                                 b,
                                                 c,
                                                 omega, 
                                                 phi), 
                                tspan, psi0, rtol=1e-7, atol=1e-7)
            UT[:,A_site_start]=sol.y[:,-1]
        
        print(time.time()-start, 'seconds.')
        """
        Plot HF
        """
        evals_U, evecs = eig(UT)
        evals_H = 1j / T *log(evals_U)
        
        HF = np.zeros([N,N], dtype=np.complex_)
        for i in range(N):
            term = evals_H[i]*np.outer(evecs[:,i], evecs[:,i])
            HF = HF+term
            
    
        ax[num0, num1].matshow(np.real(HF), interpolation='none', cmap='PuOr', norm=norm)
        #ax.set_title('real')
        ax[num0, num1].tick_params(axis="x", bottom=True, top=False,  labelbottom=True, 
          labeltop=False)
        ax[num0, num1].set_title(titles[num0*3 + num1])
        ax[num0,num1].set_xlabel('m')
        ax[num0,num1].set_ylabel('n', rotation=0, labelpad=10)
        

cax = plt.axes([1.04, 0.05, 0.04, 0.9])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
#fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm))
#fig.suptitle('FH, '+
#             'oscillating V ('+
#             'N='+str(N)+
#             ', o='+str(centre)+
#              ', a='+str(a)+
#              ', omega='+str(("{:.2f}".format(omega)))+
#              ', phi='+str("{:.1f}".format(phi/pi))+
#              'pi'
#              ')', fontsize=16)
#             
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#        'first_year_report/HF,F=30,arrangement.pdf', 
#        format='pdf', bbox_inches='tight')
plt.show()


#%%

'''
Gaus subtract time average..
'''

norm = col.Normalize(vmin=-1, vmax=1) 
N = 31; 
centre=15;
a=30;
b = 0.1;
c = 0.1;
phi=0; 

sz = 6
fig, ax = plt.subplots(figsize=(sz,sz/1.42))
    

titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
#for num0, c in enumerate([0.1]):
#    for num1, omega in enumerate([5, 8, 12]):
    
T=2*pi/omega
tspan = (0,T)
    
UT = np.zeros([N,N], dtype=np.complex_)
start = time.time()
for A_site_start in range(N):
#    print(A_site_start)
    psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
    sol = solve_ivp(lambda t,psi: F_MGSTA(t, psi, N, 
                                         centre,
                                         a, 
                                         b,
                                         c,
                                         omega, 
                                         phi), 
                        tspan, psi0, rtol=1e-7, atol=1e-7)
    UT[:,A_site_start]=sol.y[:,-1]

print(time.time()-start, 'seconds.')
"""
Plot HF
"""
evals_U, evecs = eig(UT)
evals_H = 1j / T *log(evals_U)

HF = np.zeros([N,N], dtype=np.complex_)
for i in range(N):
    term = evals_H[i]*np.outer(evecs[:,i], evecs[:,i])
    HF = HF+term
    

ax.matshow(np.real(HF), interpolation='none', cmap='PuOr', norm=norm)
#ax.set_title('real')
ax.tick_params(axis="x", bottom=True, top=False,  labelbottom=True, 
  labeltop=False)
ax.set_title(titles[num0*3 + num1])
ax.set_xlabel('m')
ax.set_ylabel('n', rotation=0, labelpad=10)
        

cax = plt.axes([1.04, 0.05, 0.04, 0.9])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
#fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm))
#fig.suptitle('FH, '+
#             'oscillating V ('+
#             'N='+str(N)+
#             ', o='+str(centre)+
#              ', a='+str(a)+
#              ', omega='+str(("{:.2f}".format(omega)))+
#              ', phi='+str("{:.1f}".format(phi/pi))+
#              'pi'
#              ')', fontsize=16)
#             
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#        'first_year_report/HF,F=30,arrangement.pdf', 
#        format='pdf', bbox_inches='tight')
plt.show()


#%%

import matplotlib


size=10
params = {
            'legend.fontsize': size*0.75,
#          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size,
          'xtick.bottom':True,
          'xtick.top':False,
          'ytick.left': True,
          'ytick.right':False,
          ## draw ticks on the left side
#          'axes.titlepad': 25
          'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
          'axes.grid':False,
          'font.family' : 'STIXGeneral',
          'mathtext.fontset':'stix'
          }

matplotlib.rcParams.update(params)


