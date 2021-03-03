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

import matplotlib as mpl
import seaborn as sns

def filter_duplicates(x):
    """
    input dataframe, df.x, eg. df.localisation
    output value 
    """
    xx = []
    # get only values
    for i in x:  #for the values in the df x
        if not np.isnan(i):
            xx.append(i)    
    if len(xx)==0:
        return np.nan
    else:
        xxx = [np.round(i, 2) for i in xx]
        if len(set(xxx))==1:
            return np.mean(xx)
        else:
            return np.nan

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

def phistring(phi):
    if phi == 0:
        return ""
    elif phi == "phi":
        return r'+ $\phi$' 
    else:
        return  r'$+ \pi /$' + str(int(1/(phi/pi)))
    
    

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



mpl.rcParams.update(params)
# plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
# plt.rcParams['grid.color'] = "0.9" # grid axis colour


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


sh = '/Users/Georgia/Code/MBQD/floquet-simulations/'

df = pd.read_csv(sh+'data/analysis-G.csv', 
                  index_col=False, 
                  converters={
                       'hopping': convert_complex,
                                'onsite':convert_complex,
                                'next onsite':convert_complex,
                                'NNN':convert_complex, 
                              'NNN overtop':convert_complex,
                                              })

df1 = pd.read_csv(sh+'data/analysis-G-py-rtol1e-9.csv', 
                  index_col=False, 
                  converters={
                        'hopping': convert_complex,
                                'onsite':convert_complex,
                                'next onsite':convert_complex,
                                'NNN':convert_complex, 
                              'NNN overtop':convert_complex,
                                              })
df2 = df1[df1.rtol==1e-9]

df = df.append(df2, ignore_index=True, sort=False)
       
        
        
# df.to_csv(sh+'data/analysis-G.csv',
#           index=False, 
#           columns=['form', 'rtol', 'a', 'omega', 'phi',
#                   'N', 'hopping', 
#                   'onsite', 'next onsite', 'NNN',
#                     'NNN overtop'])

#%%                           
"""
Plot General
"""

N = 51; 

forms=[
        # 'SS-m',
        # 'SS-p',
        # 'linear-m',
        "linear"
       ]

rtols=[ 1e-9]
aas = [35]
# phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
phis =  [0]
apply = [np.abs, np.real, np.imag]


look = 'hopping'
look = 'onsite'
look = 'next onsite'
look = 'NNN'
# look = 'NNN overtop'
# look = 'localisation'

title, indices = formatplot(look)


labels = [r'$|$'+indices+r'$|$', 
          r'$\mathrm{Real} \{$'+indices+r'$\}$',
          r'$\mathrm{Imag} \{$'+indices+r'$\}$']

    
sz =17
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
                                      + ' rtol='+str(rtol)
                                     # color='Blue'
                                    )
                        ax[n1].set_xlabel(r'$\omega$')
                        ax[n1].set_title(labels[n1])



# if form == 'OSC':
#     title1 = 'Old Numerical'
# elif form == 'OSC_conj':
#     title1 = 'Updated Numerical'
# if form == 'theoretical' or form == 'theoretical_hermitian':
#     title1 = 'Theoretical'
if form == 'SS-m':
    title1 = "Mathematica, SS"
elif form == 'SS-p':
    title1 = "Python, SS" 
elif form == 'linear':
    title1 = "Python, full chain"
elif form == 'linear-m':
    title1 = "Mathematica, full chain"
else:
    ValueError


if form in ['SS-m', "SS-p"]:
    eq =  r"$ |25><25| $"
elif form in ["linear-m", "linear-p", "linear"]:
    eq = r"$\sum_j  |j><j| j $"
else:
    ValueError
    
    
handles, labels = ax[1].get_legend_handles_labels()    
fig.legend(handles, labels, loc='upper right')
fig.suptitle(title1+', '
             + title + ' (' +indices+')'
             + r', $V(t) = $'
             +eq
             + str(a)
             +r'$ \cos( \omega t$'
             + phistring("phi")
             + r'$) $'
             + ', rtol = '+str(rtol)
             , fontsize = 20)

# fig.savefig(sh+'graphs/test.png', 
#             format='png', bbox_inches='tight')

plt.grid(True)
plt.show()



                                                         
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              

