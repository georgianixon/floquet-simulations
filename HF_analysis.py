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
              'dodgerblue']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


#%%

import matplotlib.pyplot as plt
from itertools import cycle

# cmap = plt.cm.get_cmap('Blues', 6)
# N = 2
# plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.viridis(np.linspace(1,0,N)))
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['0.8', 'green', 'blue'], 
#                                              marker=['o', '+', 'x'],
#                                              markersize=[3,2,1])


colour_index = cycle(['red', 'green', '0.3'])
markersize_index = cycle([3, 2, 1])
marker_index = cycle(['o', '+', 'x'])

#%%
sh = '/Users/Georgia/Code/MBQD/floquet-simulations/'

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
forms=['OSC']
# rtols=[1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
rtols=[1e-7]
aas = [35]
bs = [np.nan]
cs = [np.nan]
phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
# phis =  [0]
# phis =  [0]
apply = [np.abs, np.real, np.imag]
labels = [r'$|H_{n,n+1}|$', 
          r'$\mathrm{Real}|H_{n,n+1}|$',
          r'$\mathrm{Imag}|H_{n,n+1}|$']

look = 'hopping'
# look = 'onsite'
# look = 'next onsite'
# look = 'NNN'
# look = 'NNN overtop'
# look = 'localisation'
    
sz =20
fig, ax = plt.subplots(nrows=1, ncols=len(apply), figsize=(sz,sz/1.62/len(apply)),
                       constrained_layout=True, sharey=True)


for form in forms:
    
    
    for a in aas: 
        for b in bs:
            for c in cs:
                for nc, phi in enumerate(phis):
                    for rtol in rtols:
                        
                        col = next(colour_index)
                        ms = next(markersize_index)
                        mark = next(marker_index)


                        for n1, f in enumerate(apply):
                        
                        
                            if form=='MG' or form =='MGSTA':
                                df_plot = df[(df['form']==form)&
                                             (df['N']==N)&
                                                  (df['a']==a)&
                                                  (df['phi']==phi)&
                                                  (df['b']==b)&
                                                  (df['c']==c)&
                                                  (df['rtol']==rtol)]
                                
                            elif form=='OSC' or form=='OSC_i' or form=='OSC_sort':
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
                            elif form == 'theoretical':
                                df_plot = df[(df['form']==form)&
                                             (df['N']==N)&
                                                  (df['a']==a)
                                                  &
                                                  (df['phi']==phi)]

                            else:
                                raise ValueError
                            
                            df_plot = df_plot.sort_values(by=['omega'])
                            
                            
                            ax[n1].plot(df_plot['omega'], f(df_plot[look]), 
                                        # markersize=ms,
                                        # marker = mark,
                                        # color=col,
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

    
handles, labels = ax[1].get_legend_handles_labels()    
fig.legend(handles, labels, loc='upper right')
# fig.suptitle(r'Tunneling matrix element ($H_{n,n+1}$), for linear offset potential, '+r'$35 \cos(\omega t + $'
#               +r'$\pi/4)$'
             # +r'$\phi)$'
             # )
fig.savefig(sh+'test.png', 
            format='png', bbox_inches='tight')

plt.grid(True)
plt.show()



#%%





#%%

sh = '/Users/Georgia/Code/MBQD/floquet-simulations/'

df = pd.read_csv(sh+'analysis_gaus_complex.csv', 
                 index_col=False, 
                 converters={
                      'hopping': convert_complex,
                               'onsite':convert_complex,
                               'next onsite':convert_complex,
                               'NNN':convert_complex, 
                              'NNN overtop':convert_complex,
                                              })

def safe_arange(start, stop, step):
    return [round(i, 2) for i in step * np.arange(start / step, stop / step)]

omegas = safe_arange(4, 19, 0.1)
phi=pi/4
look = 'hopping'
f = np.abs


sz = 10

for omega in omegas:
    fig, ax = plt.subplots(figsize=(sz,sz/2))   
    df_plot = df[(df['form']=='linear')&
       (df['omega']==omega)&
       (df['phi']==phi)]
    
    df_plot = df_plot.sort_values(by=['rtol'])
    
    # 
    # df_plot = df_plot.sort_values(by=['rtol'], ascending=False)
    
    ax.plot(df_plot['rtol'], f(df_plot[look]), 
                label=r' $\omega=$'+"{0:.1f}".format(omega)
                # +r', $\phi=\pi /$'+"{0:.0f}".format(1/(phi/pi))
                )
    
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel(r'rtol')
    # ax.set_title(look+' element')
    fig.suptitle('Tunneling matrix element for linear offset, '+r'$35 \cos(\omega t + \pi/4)$')
    ax.invert_xaxis()
    ax.set_ylim([0,0.41])
    ax.legend()


# fig.savefig(sh+'linear_offset_pi4_rtols_freq=7.7.pdf', 
#             format='pdf', bbox_inches='tight')
    plt.show()

                                           
                                                              
                               #%%

df['rtol']=1e-7

df.loc[df['form'] == 'linear-1e-6', 'rtol'] = 1e-6
df.loc[df['form'] == 'linear-1e-8', 'rtol'] = 1e-8
df.loc[df['form'] == 'linear-1e-9', 'rtol'] = 1e-9
df.loc[df['form'] == 'linear-1e-10', 'rtol'] = 1e-10


df.loc[df['form'] == 'linear-1e-10', 'form'] = 'linear'
df.loc[df['form'] == 'linear-1e-9', 'form'] = 'linear'
df.loc[df['form'] == 'linear-1e-8', 'form'] = 'linear'
df.loc[df['form'] == 'linear-1e-6', 'form'] = 'linear'

                                             
df.to_csv(sh+'analysis_gaus_complex.csv',
                          index=False, 
                          columns=['form', 'rtol', 'a','b', 'c', 'omega', 'phi',
                                  'N', 'localisation', 'hopping', 
                                  'onsite', 'next onsite', 'NNN',
                                    'NNN overtop'])                                                       
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              

