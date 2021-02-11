# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:00:16 2020

@author: Georgia
"""
from numpy.linalg import eig
import matplotlib.colors as col

from numpy import exp, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import time
from hamiltonians import F_OSC, create_HF

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



#%%

'''HF'''
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
form = 'MG'


sz = 6
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(sz,sz/1.42), constrained_layout=True)
    

titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for num0, c in enumerate([0.1, 1]):
    for num1, omega in enumerate([5, 8, 12]):
        
        UT, HF = create_HF(form, N, centre, a, b, c, phi, omega)
    
        ax[num0, num1].matshow(np.real(HF), interpolation='none', cmap='PuOr', norm=norm)
        #ax.set_title('real')
        ax[num0, num1].tick_params(axis="x", bottom=True, top=False,  labelbottom=True, 
          labeltop=False)
        ax[num0, num1].set_title(titles[num0*3 + num1])
        ax[num0,num1].set_xlabel('m')
        ax[num0,num1].set_ylabel('n', rotation=0, labelpad=10)
        

cax = plt.axes([1.04, 0.05, 0.04, 0.9])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
#             
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#        'first_year_report/HF,F=30,arrangement.pdf', 
#        format='pdf', bbox_inches='tight')
plt.show()


#%%
''' DENSITY EVOLUTION'''

'''
Put in report
'''
import numpy as np

N = 51; A_site_start = 25;
a = 30; 
omega=7; T=2*pi/omega
phis = [0,pi/2]
centre = 25;
tspan = (0,10)
Nt = 100
sz = 14
fig, ax = plt.subplots(nrows=len(phis), ncols=3, sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))

titles = ['(a)\n', '(b)\n', '(c)\n', '(d)\n', '(e)\n', '(f)\n', ]
for nn, phi in enumerate(phis):
    t_eval = np.linspace(tspan[0], tspan[1], Nt)
    
    psi0 = np.zeros(N, dtype=np.complex_); psi0[A_site_start] = 1;
    
    sol = solve_ivp(lambda t,psi: F_OSC(t, psi, N, centre,
                                                         a,
                                                         omega, phi), 
                                        tspan, psi0, rtol=1e-7, atol=1e-7,
                                        t_eval=t_eval, method='RK45')
    
    #sz = 7
    sz = 12

    ax[nn,0].matshow(abs(sol.y)**2, interpolation='none', cmap='Purples')
    ax[nn,1].matshow(np.real(sol.y), interpolation='none', cmap='Purples')
    ax[nn,2].matshow(np.angle(sol.y), interpolation='none', cmap='Purples')
    ax[nn,0].set_title(titles[nn*3]+r'$|\psi(t)|^2$')
    ax[nn,1].set_title(titles[nn*3+1]+r'$\mathrm{Re}\{\psi(t)\}$')
    ax[nn,2].set_title(titles[nn*3+2]+r'$\mathrm{Imag}\{\psi(t)\}$')
    #ax[3].set_title('phase')
    x_positions = np.arange(0, Nt, T*(Nt/tspan[1]))
    x_labels = list(range(len(x_positions)))
    for i in range(3):
        ax[nn,i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[nn, i].set_xticks(x_positions)
        ax[nn,i].set_xlabel('T')
        ax[nn,i].set_xticklabels(x_labels)
        if i == 0:
            ax[nn,i].set_ylabel('site')
    
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.01, 0.7])
fig.colorbar(plt.cm.ScalarMappable(cmap='Purples'), cax=cbar_ax)# shrink=.5, pad=.01, aspect=10)
#fig.suptitle('F = '+str(a)+', omega='+str(omega)+ ', phi='+str(phi))
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#        'first_year_report/densityevolution,F=30,w=7p83.pdf', 
#        format='pdf', bbox_inches='tight')
plt.show()
    

#%%
'''freq vs matrix elements'''



import seaborn as sns
"""
Plot OSC
"""

df = pd.read_csv('/Users/Georgia/Code/MBQD/lattice-simulations/analysis_OSC.csv', 
                 index_col=False)

#colours = sns.color_palette("magma", 9)
#colours = sns.color_palette("Purples", 9)
colours = sns.color_palette("Greys", 8)

#plt.figure(figsize=(14,7))
sz = 6
fig, ax = plt.subplots(figsize=(sz,3.2))
c = 1

for a in [25]:
    N = 51
    df_plot = df[(df['a']==a)&(df['N']==N)]
    for cn, phi in enumerate([pi/2, pi/3, pi/4, pi/5, pi/6, pi/7,0]):
        
        cn = cn
        df_plot = df[(df['a']==a) &(df['phi']==phi)]
        df_plot = df_plot.sort_values(by=['omega'])

        look='localisation'
        look = 'hopping'
#                look = 'onsite'
#                look = 'next onsite'
#                look = 'NNN'
#                look = 'NNN overtop'
#
        

        ax.plot(df_plot['omega'], df_plot[look], color=colours[cn],
                 label=r'$\phi=$'+str(round(phi/pi, 2))+r'$\pi$')
    
#    ax.vlines(a/jn_zeros(0,4), -0.4, 0.4, 
#               colors='0.5', linestyles='dotted')
#    ax.vlines(a/jn_zeros(1,4), -0.4, 0.4, 
#               colors='r', linestyles='dotted')
#    ax.hlines([0], df_plot['omega'].min(), df_plot['omega'].max(),
#               colors='g', linestyles='dotted')
    

        roundd = lambda t: round(t, 2)
        turningvals = np.array(list(map(roundd, np.append(a/jn_zeros(0, 3), 
                                                          (a/jn_zeros(1, 3))))))
        ax.set_xticks(turningvals[turningvals>4])

ax.plot(df_plot['omega'], [jv(0, a/i) for i in df_plot['omega']], ':',
                     label=r'$\mathcal{J}_0 \left( \frac{F}{\omega} \right)$', color='0.6')
ax.set_xlabel(r'$\omega$')
#ax.set_title('Tunneling matrix element '+r"$G_{n', n'+1}$"+' ('+r'$F=$'+str(a)+
#          ')')
ax.legend()
ax.set_xlim(xmin=3.7)

#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#            'first_year_report/tunnelingmatrixelementsF=30.pdf', 
#            format='pdf', bbox_inches='tight')


plt.show()

#%%
'''
WHAT WE NEED FOR GAUS
'''


import matplotlib
import seaborn as sns
# sns.set(style="darkgrid")
# sns.set(rc={'axes.facecolor':'0.96'})
size=14
params = {
            'legend.fontsize': size*0.75,
#          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size*1.2,
          'font.family': 'STIXGeneral',
#          'axes.titlepad': 25,
          'mathtext.fontset': 'stix'
          }
matplotlib.rcParams.update(params)


CB91_Blue = 'darkblue'#'#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet, 'darkgoldenrod']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


#colours = sns.color_palette("Set2", 7)
df = pd.read_csv('/Users/Georgia/Code/MBQD/lattice-simulations/analysis_gaus.csv', 
                 index_col=False)

#plt.figure(figsize=(14,7))
sz = 7
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(sz,sz/1.42), constrained_layout=True)
c = 1
a = 30
N = 51
b = 0.1
titles = [['(a)', '(b)'],[ '(c)', '(d)']]
for n1, look in enumerate(['hopping', 'onsite']):
    for n2, c in enumerate([0.1, 1]):       
        for nc, phi in enumerate([0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]):
            
            df_plot = df[(df['a']==a) &(df['phi']==phi)&
                         (df['b']==b)&(df['c']==c)]
            df_plot = df_plot.sort_values(by=['omega'])
    
            ax[n1, n2].plot(df_plot['omega'], df_plot[look], '.',
                     label=r'$\phi=$'+str(round(phi/pi, 2))+r'$\pi$', markersize=1.6)
            
            ax[n1,n2].set_xlabel(r'$\omega$')
#            ax[n1,n2].legend()
            ax[n1,n2].set_xlim(xmin=3.7)
            ax[n1,n2].set_title(titles[n1][n2])

handles, labels = ax[0,0].get_legend_handles_labels()    
fig.legend(handles, labels, loc='best')
##
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#            'first_year_report/MG,onsite,tunneling,a=30.pdf', 
#            format='pdf', bbox_inches='tight')


plt.show()
