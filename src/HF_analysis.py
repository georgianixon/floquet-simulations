# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:47:31 2020

@author: Georgia
"""

"""
Create csv that gives hopping as a function of a, omega, type of hamiltonian,
and other parameters
"""

import matplotlib.colors as col
norm = col.Normalize(vmin=-1, vmax=1) 
from numpy import  pi, log
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv, jn_zeros
import pandas as pd 

def convert_complex(s):
    return np.complex(s.replace('i', 'j'))


def formatplot(look):
        
    if look == 'hopping':
        indices = r'$G_{n, n+1}$'
        title = 'Hopping'
    elif look == 'onsite':
        indices =  r'$G_{n, n}$'
        title = 'Onsite'
    elif look == 'next onsite':
        indices =  r'$G_{n+1, n+1}$'
        title = 'Next onsite'
    elif look == 'NNN':
        indices =  r'$G_{n, n+2}$'
        title = 'Next nearest neighbour'
    elif look == 'NNN overtop':
        indices =  r'$G_{n-1, n+1}$'
        title = 'Next nearest neighbour overtop'
    
    return  title, indices


import matplotlib
import seaborn as sns
sns.set(style="darkgrid")
sns.set(rc={'axes.facecolor':'0.96'})
size=17
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

# plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
# plt.rcParams['grid.color'] = "0.9" # grid axis colour

#%%

CB91_Blue = 'darkblue'#'#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
               CB91_Purple,
                # CB91_Violet,
                'dodgerblue',
                'slategrey',
              
              'khaki']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


#%%
sh = '/Users/Georgia/Code/MBQD/floquet-simulations/'

df = pd.read_csv(sh+'data/analysis_gaus_complex.csv', 
                 index_col=False, 
                 converters={
                      'hopping': convert_complex,
                               'onsite':convert_complex,
                               'next onsite':convert_complex,
                               'NNN':convert_complex, 
                              'NNN overtop':convert_complex,
                                              })
"""
Plot General
"""

N = 51; 
forms=[
        # 'theoretical',
        # 'OSC_conj', 
        'OSC'
       ]
# rtols=[1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
rtols=[1e-7]
aas = [35]
bs = [np.nan]
cs = [np.nan]
phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
# phis =  [0]
# phis =  [0]
apply = [np.abs, np.real, np.imag]


look = 'hopping'
look = 'onsite'
look = 'next onsite'
look = 'NNN'
look = 'NNN overtop'
# look = 'localisation'

title, indices = formatplot(look)

labels = [r'$|$' + indices + r'$|$', 
          r'$\mathrm{Real}|$'+indices+r'$|$',
          r'$\mathrm{Imag}|$'+indices+r'$|$']


    
sz =20
fig, ax = plt.subplots(ncols=len(apply), nrows=1, figsize=(sz,sz/len(apply)/1.62*0.89),
                       constrained_layout=True, sharey=True)


for form in forms:
    
    
    for a in aas: 
        for b in bs:
            for c in cs:
                for nc, phi in enumerate(phis):
                    for rtol in rtols:

                        for n1, f in enumerate(apply):
                        
                        
                            if form=='MG' or form =='MGSTA':
                                df_plot = df[(df['form']==form)&
                                             (df['N']==N)&
                                                  (df['a']==a)&
                                                  (df['phi']==phi)&
                                                  (df['b']==b)&
                                                  (df['c']==c)&
                                                  (df['rtol']==rtol)]
                                
                            elif form=='OSC' or form=='OSC_conj':
                                df_plot = df[(df['form']==form)&
                                             (df['N']==N)&
                                                  (df['a']==a) &
                                                  (df['phi']==phi)&
                                                  (df['rtol']==rtol)]
                                
                            elif form == 'linear':
                                df_plot = df[(df['form']==form)&
                                             (df['N']==N)&
                                                  (df['a']==a)
                                                  &
                                                  (df['phi']==phi)&
                                                  (df['rtol']==rtol)]
                            elif form == 'theoretical' or form == 'theoretical_hermitian':
                                df_plot = df[(df['form']==form)&
                                             (df['N']==N)&
                                                  (df['a']==a)
                                                  &
                                                  (df['phi']==phi)]

                            else:
                                raise ValueError
                            
                            df_plot = df_plot.sort_values(by=['omega'])
                            
                            
                            ax[n1].plot(df_plot['omega'], f(df_plot[look]), 
                                        label=
                                           r'$\phi=$'+str(round(phi/pi, 2))+r'$\pi$'
                                            # +', '+
                                            # form
                                         # 'rtol='+str(rtol)
                                         # color='Blue'
                                        )
                            # if  not local_n:
                            #     ax[n1].plot(df_plot['omega'], df_plot['localisation'],'.', lw=1,
                            #               label=r'localisation')
                            #     local_n = 1
                            ax[n1].set_xlabel(r'$\omega$')
                            # ax[n1].set_ylabel(r'Renormalised Tunneling')
                            # ax[n1].set_xlim(xmin=3.7)
                            
                            ax[n1].set_title(labels[n1])
                            
                            #set x points
                            # roundd = lambda t: round(t, 2)
                            # turningvals = np.array(list(map(roundd, np.append(a/jn_zeros(0, 3), 
                            #                                                   (a/jn_zeros(1, 3))))))
                            # ax.set_xticks(turningvals[turningvals>4])
                            # ax[n1].vlines(a/jn_zeros(0,4), -0.4, 0.4, colors='0.5', linestyles='dotted')
                            # ax[n1].vlines(a/jn_zeros(1,4), -0.4, 0.4,  colors='r', linestyles='dotted')
                            # extraticks = [7.5]
                            # ax.set_xticks(list(ax.get_xticks()) + extraticks)
                            # ax[n1].vlines([7.5], -0.4, 0.4,  colors='0.9', linestyles='dotted')

# # make <0 grey
# plt.axhspan(-0.4, 0, facecolor='0.4', alpha=0.5)


if form == 'OSC':
    title1 = 'Old Numerical'
elif form == 'OSC_conj':
    title1 = 'Updated Numerical'
elif form == 'theoretical' or form == 'theoretical_hermitian':
    title1 = 'Theoretical'

             
handles, labels = ax[1].get_legend_handles_labels()    
fig.legend(handles, labels, loc='upper right')
fig.suptitle(title1+', '
             + title + ' (' +indices+')'
             # +'\n' 
    + r', $V(t) = $'+
    # str(a)+r'$ \cos( \omega t)$'
      # str(a)+r'$ \cos( \omega t + \pi /$' + str(int(1/(phi/pi))) + ')'
      str(a)+r'$ \cos( \omega t + \phi)$'
    , fontsize = 20)

# fig.suptitle(r'Tunneling matrix element ($H_{n,n+1}$), for linear offset potential, '+r'$35 \cos(\omega t + $'
#               +r'$\pi/4)$'
#               +r'$\phi)$'
#               )
# fig.savefig(sh+'graphs/test.png', 
#             format='png', bbox_inches='tight')

plt.grid(True)
plt.show()



                                                         
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              

