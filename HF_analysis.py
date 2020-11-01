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



#%%
sh = '/Users/Georgia/Code/MBQD/lattice-simulations/'

df = pd.read_csv(sh+'analysis_gaus_complex.csv', 
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
form='OSC_i'
aas = [25]
bs = [np.nan]
cs = [np.nan]
phis =  [0, pi/7,pi/6, pi/5, pi/4, pi/3, pi/2]
apply = [np.abs, np.real, np.imag]
look = 'hopping'
# look = 'onsite'
# look = 'next onsite'
# look = 'NNN'
# look = 'NNN overtop
    
sz = 20
fig, ax = plt.subplots(nrows=1, ncols=len(apply), figsize=(sz,sz/5),
                       constrained_layout=True)

for n1, f in enumerate(apply):
    for a in aas: 
        for b in bs:
            for c in cs:
                for nc, phi in enumerate(phis):
                    if form=='MG' or form =='MGSTA':
                        df_plot = df[(df['form']==form)&
                                     (df['N']==N)&
                                          (df['a']==a) &
                                          (df['phi']==phi)&
                                          (df['b']==b)&
                                          (df['c']==c)]
                    if form=='OSC' or form=='OSC_i' or form=='OSC_sort':
                        df_plot = df[(df['form']==form)&
                                     (df['N']==N)&
                                          (df['a']==a) &
                                          (df['phi']==phi)]
    
                    df_plot = df_plot.sort_values(by=['omega'])
                    
                    
                    ax[n1].plot(df_plot['omega'], f(df_plot[look]), '.', lw=1,
                             label=r'$\phi=$'+str(round(phi/pi, 2))+r'$\pi$')
                    ax[n1].set_xlabel(r'$\omega$')
                    ax[n1].set_xlim(xmin=3.7)
                    
                    ax[n1].set_title(f)
                    
                    #set x points
                    roundd = lambda t: round(t, 2)
                    turningvals = np.array(list(map(roundd, np.append(a/jn_zeros(0, 3), 
                                                                      (a/jn_zeros(1, 3))))))
                    ax[n1].set_xticks(turningvals[turningvals>4])
                    # ax[n1].vlines(a/jn_zeros(0,4), -0.4, 0.4, colors='0.5', linestyles='dotted')
                    # ax[n1].vlines(a/jn_zeros(1,4), -0.4, 0.4,  colors='r', linestyles='dotted')
        

handles, labels = ax[0].get_legend_handles_labels()    
fig.legend(handles, labels, loc='right')
fig.suptitle('type='+form+',  matrix elem='+look+
             ',  a='+str(a)+',  b='+str(b)+',  c='+str(c), fontsize=16)

#fig.savefig('/Users/Georgia/Dropbox/phd/own_notes/'+
#            'first_year_report/MGtunneling,a=30,b=0p1,c=1.pdf', 
#            format='pdf', bbox_inches='tight')

plt.show()



