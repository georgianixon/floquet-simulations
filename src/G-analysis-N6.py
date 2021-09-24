# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:34:11 2021

@author: Georgia Nixon
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
from hamiltonians import  hoppingHF, ConvertComplex, PhiString



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
          'mathtext.fontset': 'stix',
          
          # 'axes.facecolor': 'white',
          'axes.edgecolor': 'white',
          'axes.grid': True,
          'grid.alpha': 1,
          # 'grid.color': "0.9"
          }


mpl.rcParams.update(params)


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
dfname = "data/analysis-G-N6.csv"
# dfname = "data/analysis-G.csv"


df = pd.read_csv(sh+dfname, 
                 index_col=False, 
                 converters={"O-3": ConvertComplex,
                            "O-2": ConvertComplex,
                            "O-1": ConvertComplex,
                            "O": ConvertComplex,
                            "O+1": ConvertComplex,
                            "O+2": ConvertComplex,
                            "O+3": ConvertComplex,
                            "N1-3": ConvertComplex,
                            "N1-2": ConvertComplex,
                            "N1-1": ConvertComplex,
                            "N1+1": ConvertComplex,
                            "N1+2": ConvertComplex,
                            "N1+3": ConvertComplex,
                            "N2-2": ConvertComplex,
                            "N2-1": ConvertComplex,
                            "N2": ConvertComplex,
                            "N2+1": ConvertComplex,
                            "N2+2": ConvertComplex,
                            "N3-2": ConvertComplex,
                            "N3-1": ConvertComplex,
                            "N3+1": ConvertComplex,
                            "N3+2": ConvertComplex,
                            "N4-1": ConvertComplex,
                            "N4": ConvertComplex,
                            "N4+1": ConvertComplex,
                            "N5-1": ConvertComplex,
                            "N5+1": ConvertComplex,
                            "N6": ConvertComplex
                            })



#%%                           
"""
Plot General
"""

N = 51; 
centre = 25



rtol=1e-11
aas = [25]
# phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
phis =  [0]
apply = [np.abs, np.real, np.imag]
omegaMax = 100
omegaMin = 6
ymax = 0.2
ymin = 0
# form = "SS-p"; hamiltonianString="$H(t)=H_0 + a \> \hat{n}_b \cos (\omega t + \phi_1) $"#; paramsString = r"$a=$"+str(a)
form = "StepFunc"; hamiltonianString="StepFunc"#; paramsString = r"$a=$"+str(a)
# form = "DS-p"; hamiltonianString = "$H(t)=H_0 + a \> \hat{n}_b \cos (\omega_1 t + \phi_1)  + a \> \hat{n}_{b+1} \cos (\omega_2 t + \phi_2)]$"; paramsString = r"$a=$"+str(a)+", "+r"$\omega_1=\omega, \omega_2 = 2 \omega, \phi_1 = \phi_1, \phi_2 = \phi_1 + \pi/2$ "
# form = "SSDF-p"; hamiltonianString = "$H(t)=H_0 + a \> \hat{n}_b [\cos (\omega_1 t + \phi_1)  +  \cos (\omega_2 t + \phi_2)]$"; paramsString = r"$a=$"+str(a)+", "+r"$\omega_1=\omega, \omega_2 = 2 \omega, \phi_1 = \phi_1, \phi_2 = \phi_1 + \pi/2$ "


termsDict = [("O-3", "G_{n-3, n-3}"),
            ("O-2","G_{n-2, n-2}"),
            ("O-1","G_{n-1, n-1}"),
            ("O","G_{n, n}"),
            ("O+1","G_{n+1, n+1}"),
            ("O+2","G_{n+2, n+2}"),
            ("O+3","G_{n+3, n+3}"),
            ("N1-3","G_{n-3, n-2}"),
            ("N1-2","G_{n-2, n-1}"),
            ("N1-1","G_{n-1, n}"),
            ("N1+1","G_{n, n+1}"),
            ("N1+2","G_{n+1, n+2}"),
            ("N1+3","G_{n+2, n+3}"),
            ("N2-2","G_{n-3, n-1}"),
            ("N2-1","G_{n-2, n}"),
            ("N2","G_{n-1, n+1}"),
            ("N2+1","G_{n, n+2}"),
            ("N2+2","G_{n+1, n+3}"),
            ("N3-2","G_{n-3, n}"),
            ("N3-1", "G_{n-2, n+1}"),
            ("N3+1", "G_{n-1, n+2}"),
            ("N3+2","G_{n, n+3}"),
            ("N4-1","G_{n-3, n+1}"),
            ("N4","G_{n-2, n+2}"),
            ("N4+1","G_{n-1, n+3}"),
            ("N5-1","G_{n-3, n+2}"),
            ("N5+1","G_{n-2, n+3}"),
            ("N6", "G_{n-3, n+3}")]
             
# look = "O-3"; matrixEl = "G_{n-3, n-3}"
# look = "O-2"; matrixEl = "G_{n-2, n-2}"
# look = "O-1"; matrixEl = "G_{n-1, n-1}"
# look = "O"; matrixEl = "G_{n, n}"
# look = "O+1"; matrixEl = "G_{n+1, n+1}"
# look = "O+2"; matrixEl = "G_{n+2, n+2}"
# look = "O+3"; matrixEl = "G_{n+3, n+3}"

# look = "N1-3"; matrixEl = "G_{n-3, n-2}"
# look = "N1-2"; matrixEl = "G_{n-2, n-1}"
# look = "N1-1"; matrixEl = "G_{n-1, n}"
# look = "N1+1"; matrixEl = "G_{n, n+1}"
# look = "N1+2"; matrixEl = "G_{n+1, n+2}"
# look = "N1+3"; matrixEl = "G_{n+2, n+3}"

# look = "N2-2"; matrixEl = "G_{n-3, n-1}"
# look = "N2-1"; matrixEl = "G_{n-2, n}"
# look = "N2"; matrixEl = "G_{n-1, n+1}"
# look = "N2+1"; matrixEl = "G_{n, n+2}"
# look = "N2+2"; matrixEl = "G_{n+1, n+3}"

# look = "N3-2"; matrixEl = "G_{n-3, n}"
# look = "N3-1"; matrixEl = "G_{n-2, n+1}"
# look = "N3+1"; matrixEl = "G_{n-1, n+2}"
# look = "N3+2"; matrixEl = "G_{n, n+3}"

# look = "N4-1"; matrixEl = "G_{n-3, n+1}"
# look = "N4"; matrixEl = "G_{n-2, n+2}"
# look = "N4+1"; matrixEl = "G_{n-1, n+3}"

# look = "N5-1"; matrixEl = "G_{n-3, n+2}"
# look = "N5+1"; matrixEl = "G_{n-2, n+3}"

# look = "N6"; matrixEl = "G_{n-3, n+3}"
        
for look, matrixEl in termsDict:

    labels = [r"$|" +matrixEl+"|$", 
              r"$\mathrm{Real} \{"+matrixEl+r"\}$",
              r"$\mathrm{Imag} \{"+matrixEl+r"\}$"]
    
    sz =15
    
    fig, ax = plt.subplots(ncols=len(apply), nrows=1, figsize=(sz,sz/len(apply)),
                           constrained_layout=True, sharey=True)
    
    
    # df.loc[df['form'] == 'SS-p','phi offset'] = np.nan
            
    for a in aas:
    
        for nc, phi in enumerate(phis):
            if form == "SS-p":
                df_plot = df[(df['form']==form)&
                         (df['N']==N)&
                          (df['a']==a)&
                          (df['phi']==phi)&
                          (df["centre"]==centre)]
                # df_plot.loc[:,"x-axis"] = df_plot.loc[:,"a"]/df_plot.loc[:,"omega"]
                
                
            # elif form =="SSDF-p" or form == "DS-p":
            #      df_plot = df[(df['form']==form)&
            #              (df['N']==N)&
            #               (df['a']==a)&
            #               (df['phi']==phi1)&
            #               (df["omega multiplier"]==omegaMultiplier)&
            #               (df['phi offset']==phiOffset)]
        
                
            if not df_plot.empty:
                # df_plot["x-axis"] = df_plot.apply(
                #     lambda row: row["a"]/row["omega"], axis=1)
                df_plot = df_plot.sort_values(by=['omega'])
                df_plot = df_plot[df_plot["omega"] < omegaMax]
                df_plot = df_plot[df_plot["omega"] > omegaMin]
                
                for n1, f in enumerate(apply):
                    ax[n1].plot(df_plot["omega"], f(df_plot[look].values), 
                                label=
                                    r'$\phi_1=$'+str(round(phi/pi, 2))+r'$\pi$'
                                    +", A="+str(a)
                                
                                    # +', '+
                                    # form
                                  # + ' rtol='+str(rtol)
                                 # color='Blue'
                                )
                    ax[n1].set_xlabel(r'$A/\omega$')
                    ax[n1].set_title(labels[n1])
        #            ax[n1].set_ylim((-0.5, 0.5))
    
        
    handles_legend, labels_legend = ax[1].get_legend_handles_labels()    
    fig.legend(handles_legend, labels_legend, loc='upper right')
    plt.grid(True)
    ax[0].set_ylim([ymin, ymax])
    fig.suptitle(""
                 + form +r";  "+hamiltonianString+"\n"
                 # +paramsString
                 +", "+look
                  # +", "+r"$a_2=$"+str(a2)
                 , y=1.2)
    plt.show()


#%%

a1 = 20
a2 = 20
#phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
phi1 =0
phiOffset = pi/2
omegaMultiplier = 2
omega1 = 10
omega2 = omega1*omegaMultiplier
phi2 = phi1+phiOffset
T = 2*pi/omega1

t = np.linspace(-2*T,2*T, 300)
y = a1*cos(omega1*t + phi1) + a2*cos(omega2*t + phi2)
plt.plot(t,y)
plt.xlabel("t")
plt.ylabel("V(t)")
plt.show()


                                                         
                                                              
                                                              
                                                              
#%%
df_plot = df[(df['form']==form)].phi1.unique()
                                                              
                                                              
                                                              
                                                   
                                                              
                                                              

