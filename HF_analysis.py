# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:47:31 2020

@author: Georgia
"""

"""
Create csv that gives hopping as a function of a, omega, type of hamiltonian,
and other parameters
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
from numpy import mean 
import time

#%%

def filter_duplicates2(x):
    xx = []
    # get only values
    for i in x:
        if not np.isnan(i):
            xx.append(i)    
    if len(xx)==0:
        return np.nan
    else:
        xxx = [round(i, 2) for i in xx]
        if len(set(xxx))==1:
            return mean(xx)
        else:
            return np.nan


def filter_duplicates(x):
    xx = [round(i, 2) for i in x]
    if len(set(xx))==1:
        return mean(x)
    else:
        return 'err'

sh = '/Users/Georgia/Code/MBQD/lattice-simulations/'
   
N = 51; 
centre=25;
form='MG' #

df = pd.read_csv(sh+'analysis_gaus_complex.csv', 
                 index_col=False)

for a in [10]:
    for b in [0.1]:
        for c in [0.1]:
            for phi in [pi/2]:
                print('a=',a,' b=',b,' c=',c,'  phi=',phi)
                df1 = pd.DataFrame(columns=['form',
                                            'a', 
                                            'b', 'c', 
                                            'omega', 'phi', 'N', 
                                            'localisation', 
                                            'hopping', 'onsite', 
                                            'next onsite', 'NNN',
                                            'NNN overtop'])
                for i, omega in enumerate(np.arange(3.7, 20, step=0.1)):
                    omega = round(omega, 1)
                    print(omega)
                    
                    """
                    HF
                    """  
                    start = time.time()
                    T=2*pi/omega
                    tspan = (0,T)
                    t_eval = np.linspace(tspan[0], tspan[1], 100)
                    UT = np.zeros([N,N], dtype=np.complex_)
                    for A_site_start in range(N):
                    #    print(A_site_start)
                        psi0 = np.zeros(N, dtype=np.complex_); 
                        psi0[A_site_start] = 1;
                        sol = solve_ivp(lambda t,psi: F_MG(t, psi, 
                                                           N, centre,
                                                             a,
                                                             b, c,
                                                             omega, phi), 
                                            tspan, psi0, rtol=1e-7, 
                                            atol=1e-7, t_eval=t_eval,
                                            method='RK45')
                        UT[:,A_site_start]=sol.y[:,-1]
                        
                    evals_U, evecs = eig(UT)
                    evals_H = 1j / T *log(evals_U)
                
                    HF = np.zeros([N,N], dtype=np.complex_)
                    for j in range(N):
                        term = evals_H[j]*np.outer(evecs[:,j], evecs[:,j])
                        HF = HF+term
                        
                    """
                    Localisation
                    """
                    psi0 = np.zeros(N, dtype=np.complex_); psi0[centre] = 1;
                    tspan = (0, 10)
                    t_eval = np.linspace(tspan[0], tspan[1], 100)
                    # single site
            #            sol = solve_ivp(lambda t,psi: F_OSC(t, psi, N,
            #                                       centre, a, omega, phi),
            #                        t_span=tspan, y0=psi0, rtol=1e-6, 
    #                                            atol=1e-6, t_eval=t_eval, 
            #                        method='RK45')
                    
                    # gaussian
                    sol = solve_ivp(lambda t,psi: F_MG(t, psi,
                                                       N, centre, 
                                                       a, b, c, 
                                                       omega, phi),
                                t_span=tspan, y0=psi0, rtol=1e-6, 
                                atol=1e-6, t_eval=t_eval, 
                                method='RK45')
                    
                    localisation = np.sum(abs(sol.y[centre]))/len(sol.t)
                    hopping=HF[centre][centre+1]
                    onsite = HF[centre][centre]
                    next_onsite=HF[centre+1][centre+1]
                    NNN = HF[centre][centre+2]
                    NNN_overtop=HF[centre-1][centre+1]
                    print('   ',time.time()-start, 's')
                    
                    # single site
            #           df1.loc[i] = [a, 
            #                   omega, phi, 
            #                   N,
            #                   localisation,
            #                   hopping,
            #                   onsite,
            #                   next_onsite,
            #                   NNN,
            #                   NNN_overtop]
                    
                    # gaussian
                    df1.loc[i] = [form, 
                           a,
                            b,
                            c,
                           omega, phi, 
                           N,
                           localisation,
                           hopping,
                           onsite,
                           next_onsite,
                           NNN,
                           NNN_overtop]
                
            
                df = df.append(df1, ignore_index=True, sort=False)
                
                df = df.groupby(['form','a', 'b', 'c', 'omega', 'phi', 
                                 'N']).agg({'localisation': filter_duplicates2,
                                        'hopping':filter_duplicates2,
                                        'onsite':filter_duplicates2,
                                        'next onsite':filter_duplicates2,
                                        'NNN':filter_duplicates2,
                                        'NNN overtop':filter_duplicates2
                                        }).reset_index()
                df.to_csv(sh+'analysis_gaus_complex.csv',
                          index=False, 
                          columns=['form', 'a','b', 'c', 'omega', 'phi',
                                   'N', 'localisation', 'hopping', 
                                   'onsite', 'next onsite', 'NNN',
                                    'NNN overtop'])
    

# single site
#df = df.groupby(['a','omega', 'phi', 
#                    'N']).agg({'localisation': filter_duplicates2,
#                            'hopping':filter_duplicates2,
#                            'onsite':filter_duplicates2,
#                            'next onsite':filter_duplicates2,
#                            'NNN':filter_duplicates2,
#                            'NNN overtop':filter_duplicates2}).reset_index()

# gaussian
#df = df.groupby(['form','a', 'b', 'c', 'omega', 'phi', 
#                 'N']).agg({'localisation': filter_duplicates2,
#                            'hopping':filter_duplicates2,
#                            'onsite':filter_duplicates2,
#                            'next onsite':filter_duplicates2,
#                            'NNN':filter_duplicates2,
#                            'NNN overtop':filter_duplicates2}).reset_index()
    
        
# single site modulation
#df.to_csv('/Users/Georgia/Code/MBQD/lattice-simulations/analysis4.csv',
#          index=False, columns=['a','omega', 'phi', 'N', 'localisation', 
#                                'hopping', 'onsite', 'next onsite', 'NNN',
#                                'NNN overtop'])
  
# gaussian     
#df.to_csv(sh+'analysis_gaus_complex.csv',
#          index=False, columns=['form','a','b', 'c', 'omega', 'phi', 'N',
#                                'localisation', 
#                                'hopping', 'onsite', 'next onsite', 'NNN',
#                                'NNN overtop'])
    

    
#%%
    
"""
Plot GAUS
"""

#colours = sns.color_palette("Set2", 7)
df = pd.read_csv('/Users/Georgia/Code/MBQD/lattice-simulations/analysis_gaus.csv', 
                 index_col=False)

#plt.figure(figsize=(14,7))
sz = 12
fig, ax = plt.subplots(figsize=(sz,sz/1.72))
c = 1
for a in [30]:
    N = 51
    df_plot = df[(df['a']==a)&(df['N']==N)]
    phi_list = df_plot.phi.unique()
    b_list = df_plot.b.unique()
    for b in [0.1]:
        for c in [1]:
            for nc, phi in enumerate([0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]):
                df_plot = df[(df['a']==a) &(df['phi']==phi)&(df['b']==b)&(df['c']==c)]
                df_plot = df_plot.sort_values(by=['omega'])
        
        #        if phi == 0:
        #            plt.plot(df_plot['omega'], df_plot['localisation'], 
        #                     label='localisation'
        #                     )
                look='localisation'
                look = 'hopping'
                look = 'onsite'
                look = 'next onsite'
#                look = 'NNN'
#                look = 'NNN overtop'
        #        
#                if b==1 and c == 0.1 and phi == 0:
#                    continue
#                if b ==1 and c == 1 and phi == 0:
#                    continue
#                if phi == 0 and b == 0.1 and c == 1:
#                    continue
                
        
                ax.plot(df_plot['omega'], df_plot[look], '.',
                         label=r'$\phi=$'+str(round(phi/pi, 2))+r'$\pi$')
            
        #    ax.vlines(a/jn_zeros(0,4), -0.4, 0.4, 
        #               colors='0.5', linestyles='dotted')
        #    ax.vlines(a/jn_zeros(1,4), -0.4, 0.4, 
        #               colors='r', linestyles='dotted')
        #    ax.hlines([0], df_plot['omega'].min(), df_plot['omega'].max(),
        #               colors='g', linestyles='dotted')
            
        #    ax.plot(df_plot['omega'], [jv(0, a/i) for i in df_plot['omega']], 
        #                     label=r'$\mathcal{J}_0 \left( \frac{F}{\omega} \right)$')
            
        
#            roundd = lambda t: round(t, 2)
#            turningvals = np.array(list(map(roundd, np.append(a/jn_zeros(0, 3), 
#                                                              (a/jn_zeros(1, 3))))))
#            ax.set_xticks(turningvals[turningvals>4])
    

ax.set_xlabel(r'$\omega$')
#ax.set_title('Tunneling matrix element '+r"$G_{n', n'+1}$"+' ('+r'$F=$'+str(a)+
#          ')')
ax.legend()
ax.set_xlim(xmin=3.7)
#
#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#            'first_year_report/MGtunneling,a=30,b=0p1,c=1.pdf', 
#            format='pdf', bbox_inches='tight')


plt.show()

#%%

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

for a in [30]:
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



#%%

import matplotlib
import seaborn as sns
#matplotlib.rcParams['mathtext.fontset'] = 'cm' #latex style, cm?
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
sns.set(style="darkgrid")
#matplotlib.rcParams["font.size"]=10
sns.set(rc={'axes.facecolor':'0.96'})
#font = {'family' : , 
#        'size'   : 10}
size=10
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
#matplotlib.rc('font', **font)

#%%

CB91_Blue = 'darkblue'#'#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet, 'darkgoldenrod']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

