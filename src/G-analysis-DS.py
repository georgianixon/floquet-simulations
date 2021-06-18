# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:34:11 2021

@author: Georgia Nixon
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:47:31 2020

@author: Georgia
"""

"""
Create csv that gives hopping as a function of a, omega, type of hamiltonian,
and other parameters
"""

place = "Georgia Nixon"
import matplotlib.colors as col
norm = col.Normalize(vmin=-1, vmax=1) 
from numpy import  pi, log
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv, jn_zeros
import pandas as pd

import matplotlib as mpl
import seaborn as sns
from numpy import sin, cos, exp, pi

import sys
sys.path.append("/Users/" + place + "/Code/MBQD/floquet-simulations/src")
from hamiltonians import  hoppingHF



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


def phistring(phi):
    if phi == 0:
        return ""
    elif phi == "phi":
        return r'+ \phi' 
    else:
        return  r'+ \pi /' + str(int(1/(phi/pi)))
    
    

sns.set(style="darkgrid")
sns.set(rc={'axes.facecolor':'0.96'})
size=20
params = {
            'legend.fontsize': size*0.75,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size,
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
red = "#FC4445"

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
               CB91_Purple,
                # CB91_Violet,
                'dodgerblue',
                'slategrey']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


sh = "/Users/" + place + "/Code/MBQD/floquet-simulations/"
dfname = "data/analysis-G-newelements-1.csv"
# dfname = "data/analysis-G.csv"

df = pd.read_csv(sh+dfname, 
                 index_col=False, 
                 converters={"square": convert_complex,
                            "chi": convert_complex,
                            "gamma": convert_complex,
                            "triangle": convert_complex,
                            "alpha": convert_complex,
                            "tilde": convert_complex,
                            "star": convert_complex,
                            "beta": convert_complex,
                            "rho": convert_complex,
                            "epsilon": convert_complex,
                            "delta": convert_complex
                            })



#%%                           
"""
Plot General
"""

N = 51; 
centre = 25
form = "DS-p"
rtol=1e-11
a1 = 35
a2 = 35
#phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
phi1s =  [0, pi/2]
phiOffset = pi/4
apply = [np.abs, np.real, np.imag]
omegaMin = 80

#look = "square"
look ="chi"
#look ="gamma"
#look ="triangle"
#look ="alpha"
#look ="tilde"
#look ="star"
#look ="beta"
#look ="rho"
#look ="epsilon"
#look ="delta"


labels = [r"$|\mathrm{"+look+"}|$", 
          r'$\mathrm{Real} \{$'+look+r'$\}$',
          r'$\mathrm{Imag} \{$'+look+r'$\}$']

sz =10

fig, ax = plt.subplots(ncols=len(apply), nrows=1, figsize=(sz,sz/len(apply)*1.6),
                       constrained_layout=True, sharey=True)


for nc, phi1 in enumerate(phi1s):
    df_plot = df[(df['form']==form)&
                 (df['N']==N)&
                  (df['a1']==a1)&
                  (df['a2']==a2)&
                  (df['phi1']==phi1)]
#                  (df['phi2']==phi1+phiOffset)

        
    if not df_plot.empty:
        print('yes')
        
        df_plot = df_plot.sort_values(by=['omega1'])
        df_plot = df_plot[df_plot["omega1"] < omegaMin]
        
        for n1, f in enumerate(apply):
            ax[n1].plot(df_plot['omega1'], f(df_plot[look].values), 
                        label=
                            r'$\phi1=$'+str(round(phi1/pi, 2))+r'$\pi$'
                            # +', '+
                            # form
                          # + ' rtol='+str(rtol)
                         # color='Blue'
                        )
            ax[n1].set_xlabel(r'$\omega$')
            ax[n1].set_title(labels[n1])
#            ax[n1].set_ylim((-0.5, 0.5))
    else:
        print('no')

    
handles_legend, labels_legend = ax[1].get_legend_handles_labels()    
fig.legend(handles_legend, labels_legend, loc='upper right')
plt.grid(True)
plt.show()


#%%

x = np.linspace(-4*pi,4*pi, 40)
y = cos(x)
y1 = cos(x+pi/7)
plt.plot(x, y+y1)


                                                         
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              
                                                              

