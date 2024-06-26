# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:52:33 2022

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
from hamiltonians import  hoppingHF, ConvertComplex, PhiString, ConvertFraction
dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/Triangle/"
from fractions import Fraction

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
size=12
params = {
            'legend.fontsize': size*0.8,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
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


# CB91_Blue = 'darkblue'#'#2CBDFE'
# CB91_Green = '#47DBCD'
# CB91_Pink = '#F3A0F2'
# CB91_Purple = '#9D2EC5'
# CB91_Violet = '#661D98'
# CB91_Amber = '#F5B14C'
# red = "#FC4445"

# color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
#                CB91_Purple,
#                 # CB91_Violet,
#                 'dodgerblue',
#                 'slategrey']
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

# sh = "/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/"
# sh = "/Users/" + place + "/Code/MBQD/floquet-simulations/"
# dfname = "analysis-G-Triangle-2Site.csv"
# dfname = "analysis-G-Triangle-2Site-RemoveGauge-1.csv"
# dfname = "analysis-G-Triangle-2Site-Full.csv"



# df = pd.read_csv(dataLoc+dfname, 
#                  index_col=False, 
#                  converters={"O-1": ConvertComplex,
#                             "O-2": ConvertComplex,
#                             "O-3": ConvertComplex,
#                             "N1-1": ConvertComplex,
#                             "N1-2": ConvertComplex,
#                             "N1-3": ConvertComplex,
#                             "timeOffset":ConvertFraction,
#                             "time1":ConvertFraction
#                             })

# dfN = df[df["omega multiplier"]!=1.5]
# df["omega multiplier"] = df["omega2"] / df["omega1"]
# df1 = df[df["omega multiplier"]==2]
# df["omega multiplier"] = df["omega multiplier"].apply(np.int64)



#%%                           
"""
Plot General
"""


phi1s =  [0]
# phi2s =  [0, pi/7, pi/6, pi/5]#, pi/4, pi/3]
# phis =  [0]
apply = [np.abs, np.real, np.imag]
omegaMax = 20
omegaMin = 4
omegaMultiplier = 2
ymax = None
ymin = None


# df["omega multiplier"] = df["omega2"]/df["omega1"]
# df["omega multiplier"] = df["omega multiplier"].astype(int)
# dfCos = df[(df["func1"]==func1Name) & (df["func2"]==func2Name)]


termsDict = [ 
    ("O-1", "G_{0, 0}"),
            ("O-2","G_{1, 1}"),
            ("O-3","G_{2, 2}"),
            ("N1-1","G_{0, 1}"),
            ("N1-2","G_{1, 2}"),
            ("N1-3","G_{0, 2}")]
            

        
for look, matrixEl in termsDict:

    labels = [r"$|" +matrixEl+"|$", 
              r"$\mathrm{Real} \{"+matrixEl+r"\}$",
              r"$\mathrm{Imag} \{"+matrixEl+r"\}$"]
    
    sz =12
    
    fig, ax = plt.subplots(ncols=len(apply), nrows=1, figsize=(sz,sz/len(apply)),
                           constrained_layout=True, sharey=True)
    
    
    # df.loc[df['form'] == 'SS-p','phi offset'] = np.nan
            
    for a in aas:
    
        # for nc, phi1 in enumerate(phi1s):
            # for nc2, phi2 in enumerate(phi2s):
        for nc, time1 in enumerate(time1s):
            for nc2, timeOffset in enumerate(timeOffsets):

                df_plot = bigData[(bigData['form']==form)&
                         (bigData['N']==N)&
                          (bigData['a1']==a)&
                          (bigData['a2']==a)&
                           # (df['phi1']==phi1)&
                           # (df['phi2']==phi2)&
                           (bigData["time1"].round(10)==time1.round(10))&
                           (bigData["timeOffset"].round(10)==timeOffset.round(10))&
                          (bitData["omega multiplier"]==omegaMultiplier)&
                          (bitData["centre1"]==centre1)&
                          (df["func1"]==func1Name)&
                          (df["func2"]==func2Name)]
                    # df_plot.loc[:,"x-axis"] = df_plot.loc[:,"a"]/df_plot.loc[:,"omega"]
                    
            
                    
                if not df_plot.empty:
                    # df_plot["x-axis"] = df_plot.apply(
                    #     lambda row: row["a"]/row["omega"], axis=1)
                    
                    
                    # df_plot = df_plot.sort_values(by=['x-axis'])
                    
                    df_plot = df_plot.sort_values(by=['omega1'])
                    df_plot = df_plot[df_plot["omega1"] < omegaMax]
                    df_plot = df_plot[df_plot["omega1"] > omegaMin]
                    
                    for n1, f in enumerate(apply):
                        
                        ax[n1].plot(df_plot["omega1"], f(df_plot[look].values), 
                                    label= (
                                        # r'$\phi1=' + PhiString(phi1) + r', \phi2=' + PhiString(phi2) +r'\pi' +
                                        r"$time_1=T*"+str(Fraction(time1).limit_denominator(24))+r", timeOffset=T*"+str(Fraction(timeOffset).limit_denominator(24))+
                                        r", A=" + str(a)
                                        +r'$'
                                        ))
                        
                        # ax[n1].plot(df_plot["x-axis"], f(df_plot[look].values), 
                        #             label=
                        #                 r'$\phi_1=$'+str(round(phi/pi, 2))+r'$\pi$'
                        #                 +", A="+str(a))
                        
                        # ax[n1].set_xlabel(r'$A/\omega$')
                        ax[n1].set_xlabel(r'$\omega$')
                        ax[n1].set_title(labels[n1])
            #            ax[n1].set_ylim((-0.5, 0.5))
        
        
    handles_legend, labels_legend = ax[1].get_legend_handles_labels()    
    fig.legend(handles_legend, labels_legend, loc='upper right')
    plt.grid(True)
    ax[0].set_ylim([ymin, ymax])
    plt.show()


#%%

      
#For paper

N = 3; 
centre1 = 1
centre2 = 2
rtol=1e-11
aas = [35]
phi1s =  [0]#[0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
phi2s =  [0]#[0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
# phis =  [0]
apply = [np.abs, np.real, np.imag]
omegaMax = 20
omegaMin = 4
ymax = None
ymin = None
form = "Tri-RemoveGauge"#"Tri"
func1Name = "Cosine"##"Blip"#"RampHalf"#
func2Name = "Cosine"##"Blip"#"RampHalf"#


termsDict = [ 
    # ("O-1", "G_{0, 0}"),
            # ("O-2","G_{1, 1}"),
            # ("O-3","G_{2, 2}"),
            ("N1-1","G_{0, 1}"),
            ("N1-2","G_{1, 2}"),
            ("N1-3","G_{0, 2}")]


look = termsDict[0][0]
matrixEl = termsDict[0][1]


sz =4

fig, ax = plt.subplots(figsize=(sz,sz),constrained_layout=True, sharey=True)


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
            

            ax.plot(df_plot["omega"], np.abs(df_plot[look].values), 
                        label=
                            r'$\phi=$'+str(round(phi/pi, 2))+r'$\pi$'
                            +", A="+str(a))
            # ax.plot(df_plot["x-axis"], np.abs((df_plot[look].values)), 
            #             label=
            #                 r'$\phi_1=$'+str(round(phi/pi, 2))+r'$\pi$'
            #                 +", A="+str(a))
            ax.set_ylabel(r"$|" +matrixEl+"|$")
            # ax.set_xlabel(r'$\frac{A}{\omega}$', fontsize=18)
            ax.set_xlabel(r'$\omega$')


    
handles_legend, labels_legend = ax.get_legend_handles_labels()    
fig.legend(handles_legend, labels_legend, loc='upper right')
plt.grid(True)
ax.set_ylim([ymin, ymax])
paper = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Paper/Figures/"
fig.savefig(paper+'G-StepFunc-elements-O-A=35-ylim=1.pdf', format='pdf', bbox_inches='tight')
plt.show()
                                                              
                                  