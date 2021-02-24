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
    return np.complex(s.replace('i', 'j').replace('*I', 'j').replace('*^', 'e'))


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
size=20
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
                'slategrey']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


#%%
sh = '/Users/Georgia/Code/MBQD/floquet-simulations/'


df = pd.read_csv(sh+'data/analysis-G-mat-lin-35.csv', 
                  index_col=False, 
                  converters={
                      'hopping': convert_complex,
                                'onsite':convert_complex,
                                'next onsite':convert_complex,
                                'NNN':convert_complex, 
                              'NNN overtop':convert_complex,
                                              })

df1 = pd.read_csv(sh+'data/analysis-G.csv', 
                  index_col=False, 
                  converters={
                      'hopping': convert_complex,
                                'onsite':convert_complex,
                                'next onsite':convert_complex,
                                'NNN':convert_complex, 
                              'NNN overtop':convert_complex,
                                              })

df = df.append(df1, ignore_index=True, sort=False)

df = df.groupby(by=['form', 'rtol', 'a', 'omega', 'phi', 
                 'N'], dropna=False).agg({
                        'hopping':filter_duplicates,
                        'onsite':filter_duplicates,
                        'next onsite':filter_duplicates,
                        'NNN':filter_duplicates,
                        'NNN overtop':filter_duplicates
                        }).reset_index()
                             
                             
"""
Plot General
"""

N = 51; 

forms=[
        # 'SS-m',
        # 'SS-p',
        'linear-m',
        "linear"
       ]

rtols=[1e-7]
aas = [35]
# phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
phis =  [0, pi/7, pi/6]
apply = [np.abs, np.real, np.imag]


look = 'hopping'
look = 'onsite'
look = 'next onsite'
# look = 'NNN'
# look = 'NNN overtop'
# look = 'localisation'

title, indices = formatplot(look)


labels = [r'$|$'+indices+r'$|$', 
          r'$\mathrm{Real} \{$'+indices+r'$\}$',
          r'$\mathrm{Imag} \{$'+indices+r'$\}$']

    
sz =15
fig, ax = plt.subplots(ncols=len(apply), nrows=1, figsize=(sz,sz/len(apply)/1.62*1.4),
                       constrained_layout=True, sharey=True)


for form in forms:
    for a in aas: 
        for nc, phi in enumerate(phis):
            for rtol in rtols:

                for n1, f in enumerate(apply):
                        
                    if form=='OSC' or form=='OSC_conj' or form =="SS-p" or form == 'linear':
                        df_plot = df[(df['form']==form)&
                                     (df['N']==N)&
                                          (df['a']==a) &
                                          (df['phi']==phi)&
                                          (df['rtol']==rtol)]
                        

                    elif form == 'theoretical' or form == 'theoretical_hermitian':
                        df_plot = df[(df['form']==form)&
                                     (df['N']==N)&
                                          (df['a']==a)
                                          &
                                          (df['phi']==phi)]
                    elif form =='OSC-mathematica'or form =="SS-m" or form == "linear-m":
                        df_plot = df[(df['form']==form)&
                                     (df['N']==N)&
                                          (df['a']==a) &
                                          (df['phi']==phi)]
                    

                    else:
                        raise ValueError
                    
                    if not df_plot.empty:
                        df_plot = df_plot.sort_values(by=['omega'])
                        
                        
                        ax[n1].plot(df_plot['omega'], f(df_plot[look]), 
                                    label=
                                       r'$\phi=$'+str(round(phi/pi, 2))+r'$\pi$'
                                        +', '+
                                        form
                                     # 'rtol='+str(rtol)
                                     # color='Blue'
                                    )
                        ax[n1].set_xlabel(r'$\omega$')
                        ax[n1].set_title(labels[n1])



if form == 'OSC':
    title1 = 'Old Numerical'
elif form == 'OSC_conj':
    title1 = 'Updated Numerical'
elif form == 'theoretical' or form == 'theoretical_hermitian':
    title1 = 'Theoretical'
elif form == 'SS-m':
    title1 = "Mathematica"
elif form == 'SS-p':
    title1 = "Python" 
elif form == 'linear':
    title1 = "Python, full chain"

             
handles, labels = ax[1].get_legend_handles_labels()    
fig.legend(handles, labels, loc='upper right')
fig.suptitle(title1+', '
              + title + ' (' +indices+')'
              # +'\n' 
    + r', $V(t) = $'+
    str(a)+r'$ \cos( \omega t)$'
       # str(a)+r'$ \cos( \omega t + \pi /$' + str(int(1/(phi/pi))) + ')'
      # str(a)+r'$ \cos( \omega t + \phi)$'
      + ', rtol = '+str(rtol)
    , fontsize = 20)

# fig.savefig(sh+'graphs/test.png', 
#             format='png', bbox_inches='tight')

plt.grid(True)
plt.show()



                                                         
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              

