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
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
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
size=10
params = {
            'legend.fontsize': size*0.7,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.7,
          'ytick.labelsize': size*0.7,
          'font.size': size,
          'font.family': 'STIXGeneral',
#          'axes.titlepad': 25,
          'mathtext.fontset': 'stix',
          
          # 'axes.facecolor': 'white',
          'axes.edgecolor': 'white',
          'axes.grid': True,
          'grid.alpha': 1,
          # 'grid.color': "0.9"
          "text.usetex": True
          }


mpl.rcParams.update(params)
mpl.rcParams["text.latex.preamble"] = mpl.rcParams["text.latex.preamble"] + r'\usepackage{xfrac}'

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


# sh = "/Users/" + place + "/Code/MBQD/floquet-simulations/"
dataLoc = "/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
dfname = "analysis-G-N6.csv"
# dfname = "data/analysis-G.csv"


df = pd.read_csv(dataLoc+dfname, 
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
aas = [5, 10, 15, 20, 25, 30, 35]
# aas = [35]
# phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
phis =  [0]
apply = [np.abs, np.real, np.imag]
omegaMax = 20
omegaMin = 4
ymax = 0.5
ymin = None
form = "SS-p"; hamiltonianString="$H(t)=H_0 + a \> \hat{n}_b \cos (\omega t + \phi_1) $"#; paramsString = r"$a=$"+str(a)
# form = "StepFunc"; hamiltonianString="StepFunc"#; paramsString = r"$a=$"+str(a)
# form = "DS-p"; hamiltonianString = "$H(t)=H_0 + a \> \hat{n}_b \cos (\omega_1 t + \phi_1)  + a \> \hat{n}_{b+1} \cos (\omega_2 t + \phi_2)]$"; paramsString = r"$a=$"+str(a)+", "+r"$\omega_1=\omega, \omega_2 = 2 \omega, \phi_1 = \phi_1, \phi_2 = \phi_1 + \pi/2$ "
# form = "SSDF-p"; hamiltonianString = "$H(t)=H_0 + a \> \hat{n}_b [\cos (\omega_1 t + \phi_1)  +  \cos (\omega_2 t + \phi_2)]$"; paramsString = r"$a=$"+str(a)+", "+r"$\omega_1=\omega, \omega_2 = 2 \omega, \phi_1 = \phi_1, \phi_2 = \phi_1 + \pi/2$ "


termsDict = [ 
    # ("O-3", "G_{n-3, n-3}"),
            # ("O-2","G_{n-2, n-2}"),
            # ("O-1","G_{n-1, n-1}"),
            # ("O","G_{n, n}"),
            # ("O+1","G_{n+1, n+1}"),
            # ("O+2","G_{n+2, n+2}"),
            # ("O+3","G_{n+3, n+3}"),
            # ("N1-3","G_{n-3, n-2}"),
            # ("N1-2","G_{n-2, n-1}"),
            ("N1-1","G_{n-1, n}"),
            # ("N1+1","G_{n, n+1}"),
            # ("N1+2","G_{n+1, n+2}"),
            # ("N1+3","G_{n+2, n+3}"),
            # ("N2-2","G_{n-3, n-1}"),
            # ("N2-1","G_{n-2, n}"),
            # ("N2","G_{n-1, n+1}"),
            # ("N2+1","G_{n, n+2}"),
            # ("N2+2","G_{n+1, n+3}"),
            # ("N3-2","G_{n-3, n}"),
            # ("N3-1", "G_{n-2, n+1}"),
            # ("N3+1", "G_{n-1, n+2}"),
            # ("N3+2","G_{n, n+3}"),
            # ("N4-1","G_{n-3, n+1}"),
            # ("N4","G_{n-2, n+2}"),
            # ("N4+1","G_{n-1, n+3}"),
            # ("N5-1","G_{n-3, n+2}"),
            # ("N5+1","G_{n-2, n+3}"),
            # ("N6", "G_{n-3, n+3}")
            ]
            
        
for look, matrixEl in termsDict:

    labels = [r"$|" +matrixEl+"|$", 
              r"$\mathrm{Real} \{"+matrixEl+r"\}$",
              r"$\mathrm{Imag} \{"+matrixEl+r"\}$"]
    
    sz =12
    
    fig, ax = plt.subplots(ncols=len(apply), nrows=1, figsize=(sz,sz/len(apply)),
                           constrained_layout=True, sharey=True)
    
    
    # df.loc[df['form'] == 'SS-p','phi offset'] = np.nan
            
    for a in aas:
    
        for nc, phi in enumerate(phis):
            if form == "SS-p" or form =="StepFunc":
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
                df_plot["x-axis"] = df_plot.apply(
                    lambda row: row["a"]/row["omega"], axis=1)
                
                
                df_plot = df_plot.sort_values(by=['x-axis'])
                
                df_plot = df_plot.sort_values(by=['omega'])
                df_plot = df_plot[df_plot["omega"] < omegaMax]
                df_plot = df_plot[df_plot["omega"] > omegaMin]
                
                for n1, f in enumerate(apply):
                    # ax[n1].plot(df_plot["omega"], f(-jv(0,a/df_plot["omega"])), label="bessel func")
                    # ax[n1].plot(df_plot["x-axis"], f(-jv(0,df_plot["x-axis"])), label="bessel func")
                    
                    ax[n1].plot(df_plot["omega"], f(df_plot[look].values), 
                                label= r'$\phi=$' + str(round(phi/pi, 2)) + r'$\pi$' + ", A=" + str(a))
                    
                    # ax[n1].plot(df_plot["x-axis"], f(df_plot[look].values), 
                    #             label=
                    #                 r'$\phi_1=$'+str(round(phi/pi, 2))+r'$\pi$'
                    #                 +", A="+str(a))
                    
                    ax[n1].set_xlabel(r'$A/\omega$')
                    ax[n1].set_xlabel(r'$\omega$')
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

      
#For paper

N = 51; 
centre = 25

rtol=1e-11
aas = [35]#[35, 30, 25, 20, 15, 10, 5]
# aas = [30]
phis =  [0]
# phis =  [0]
apply = [np.abs]
omegaMax = 20
omegaMin = 4
ymax = None
ymin = None
form = "SS-p";
# form = "StepFunc"; 
# form = "DS-p"; 
# form = "SSDF-p"; 

termsDict = [ 
    # ("O-3", "G_{n-3, n-3}"),
            # ("O-2","G_{n-2, n-2}"),
            # ("O-1","G_{n-1, n-1}"),
            # ("O","G_{n, n}"),
            # ("O+1","G_{n+1, n+1}"),
            # ("O+2","G_{n+2, n+2}"),
            # ("O+3","G_{n+3, n+3}"),
            # ("N1-3","G_{n-3, n-2}"),
            # ("N1-2","G_{n-2, n-1}"),
            # ("N1-1","G_{n-1, n}"),
            ("N1+1","G_{b, b+1}"),
            # ("N1+2","G_{n+1, n+2}"),
            # ("N1+3","G_{n+2, n+3}"),
            # ("N2-2","G_{n-3, n-1}"),
            # ("N2-1","G_{n-2, n}"),
            # ("N2","G_{n-1, n+1}"),
            # ("N2+1","G_{n, n+2}"),
            # ("N2+2","G_{n+1, n+3}"),
            # ("N3-2","G_{n-3, n}"),
            # ("N3-1", "G_{n-2, n+1}"),
            # ("N3+1", "G_{n-1, n+2}"),
            # ("N3+2","G_{n, n+3}"),
            # ("N4-1","G_{n-3, n+1}"),
            # ("N4","G_{n-2, n+2}"),
            # ("N4+1","G_{n-1, n+3}"),
            # ("N5-1","G_{n-3, n+2}"),
            # ("N5+1","G_{n-2, n+3}"),
            # ("N6", "G_{n-3, n+3}")
            ]


look = termsDict[0][0]
matrixEl = termsDict[0][1]


sz =3.2

fig, ax = plt.subplots(figsize=(sz,0.6*sz),constrained_layout=True, sharey=True)


# df.loc[df['form'] == 'SS-p','phi offset'] = np.nan
        
for a in aas:

    for nc, phi in enumerate(phis):
        if form == "SS-p" or form =="StepFunc":
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
            df_plot["x-axis"] = df_plot.apply(
                lambda row: row["a"]/row["omega"], axis=1)
            
            
            df_plot = df_plot.sort_values(by=['x-axis'])
            
            # df_plot = df_plot.sort_values(by=['omega'])
            # df_plot = df_plot[df_plot["omega"] < omegaMax]
            # df_plot = df_plot[df_plot["omega"] > omegaMin]
            

            # ax.plot(df_plot["omega"], np.real(df_plot[look].values), 
            #             label=
            #                 r'$\phi=$'+str(round(phi/pi, 2))+r'$\pi$'
            #                 +", A="+str(a))
            ax.plot(df_plot["x-axis"], -np.real((df_plot[look].values)), 
                        label=r"$J' = -" +matrixEl+"$")
            # ax.set_ylabel()
            ax.set_xlabel(r'$\sfrac{A}{\omega}$', fontsize=13)
            # ax.set_xlabel(r'$\omega$')
        ax.plot(df_plot["x-axis"], jv(0, df_plot["x-axis"]), '--', 
                        label=r"$\mathcal{J}_0 (A / \omega)$")

    
handles_legend, labels_legend = ax.get_legend_handles_labels()    
fig.legend(handles_legend, labels_legend, loc='upper right')
plt.grid(True)
ax.set_ylim([ymin, ymax])
paper = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Paper/Figures/"
# fig.savefig(paper+'ElementG-SS-Tunnelling.pdf', format='pdf', bbox_inches='tight')
plt.show()
                                                              
   #%%

"""
for poster
"""

N = 51; 
centre = 25



rtol=1e-11
a = 35
# aas = [35]
# phis =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
phi =  0
apply = [np.abs, np.real, np.imag]
omegaMax = 20
omegaMin = 4
ymax = 0.5
ymin = None
form = "SS-p";

termsDict = [ 
            ("N1-1","G_{n-1, n}"),
            ]
            
        
for look, matrixEl in termsDict:
    
    sz =3
    
    fig, ax = plt.subplots( figsize=(sz,sz/2),
                           constrained_layout=True, sharey=True)
    

    df_plot = df[(df['form']==form)&
             (df['N']==N)&
              (df['a']==a)&
              (df['phi']==phi)&
              (df["centre"]==centre)]
    # df_plot.loc[:,"x-axis"] = df_plot.loc[:,"a"]/df_plot.loc[:,"omega"]
        

        
    if not df_plot.empty:
        df_plot["x-axis"] = df_plot.apply(
            lambda row: row["a"]/row["omega"], axis=1)
        
        
        df_plot = df_plot.sort_values(by=['x-axis'])
        
        df_plot = df_plot.sort_values(by=['omega'])
        df_plot = df_plot[df_plot["omega"] < omegaMax]
        df_plot = df_plot[df_plot["omega"] > omegaMin]

            
        ax.plot(df_plot["omega"], np.real(df_plot[look].values), 
                    label= r'$\phi=$' + str(round(phi/pi, 2)) + r'$\pi$' + ", A=" + str(a), color="#FF0000")
        
        # ax.plot(df_plot["x-axis"], np.real(df_plot[look].values), 
        #             label=
        #                 r'$\phi_1=$'+str(round(phi/pi, 2))+r'$\pi$'
        #                 +", A="+str(a), color="r")
        
        # ax.set_xlabel(r'$A/\omega$')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r"$J'$", rotation=0, color="#FF0000", labelpad=15)
#            ax[n1].set_ylim((-0.5, 0.5))
    
        
    handles_legend, labels_legend = ax.get_legend_handles_labels()    
    # fig.legend(handles_legend, labels_legend, loc='upper right')
    plt.grid(True)
    ax.set_ylim([ymin, ymax])
    paper = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Paper/Figures/"
    # fig.savefig(paper+'ElementG-Tunnelling-A=35-poster.pdf', format='pdf', bbox_inches='tight')
    plt.show()                                                           

