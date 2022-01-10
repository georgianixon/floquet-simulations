# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:33:28 2022

@author: Georgia
"""
""" check which one has got hoppings most similar and onsite energies closes to zero"""

place = "Georgia"
import sys
import pandas as pd
sys.path.append("/Users/" + place + "/Code/MBQD/floquet-simulations/src")
from hamiltonians import  hoppingHF, ConvertComplex, PhiString
import numpy as np


sh = "/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/"
# sh = "/Users/" + place + "/Code/MBQD/floquet-simulations/"
# dfname = "analysis-G-Triangle-2Site.csv"
dfname = "analysis-G-Triangle-2Site-RemoveGauge.csv"
# dfname = "analysis-G-Triangle-2Site-Full.csv"
dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
df = pd.read_csv(dataLoc+dfname, 
                 index_col=False, 
                 converters={"O-1": ConvertComplex,
                            "O-2": ConvertComplex,
                            "O-3": ConvertComplex,
                            "N1-1": ConvertComplex,
                            "N1-2": ConvertComplex,
                            "N1-3": ConvertComplex,
                            })








dfr = df[df["form"]=="Tri-RemoveGauge"]

#get rows with same abs val of hopping
spread = 0.035
df["abs(O-1)"] = np.abs(df["O-1"])
df["abs(O-2)"] = np.abs(df["O-2"])
df["abs(O-3)"] = np.abs(df["O-3"])
df["abs(N1-1)"] = np.abs(df["N1-1"])
df["abs(N1-2)"] = np.abs(df["N1-2"])
df["abs(N1-3)"] = np.abs(df["N1-3"])
df["abs(N1-1)-abs(N1-2)"] = np.abs(df["abs(N1-1)"]-df["abs(N1-2)"])
df["abs(N1-1)-abs(N1-3)"] = np.abs(df["abs(N1-1)"]-df["abs(N1-3)"])
df["abs(N1-2)-abs(N1-3)"] = np.abs(df["abs(N1-2)"]-df["abs(N1-3)"])

#find which are negative and positie
df["sign(N1-1)"] = np.sign(np.real(df['N1-1']))
df["sign(N1-2)"] = np.sign(np.real(df['N1-2']))
df["sign(N1-3)"] = np.sign(np.real(df['N1-3']))
df["total sign"] = df["sign(N1-1)"]+df["sign(N1-2)"]+df["sign(N1-3)"]


dfn = df[(df["total sign"] != -3) & (df["total sign"]!=1) # make sure only one flip or 3 flilp
                 &
         (df["abs(N1-1)-abs(N1-2)"]<=spread)&
         (df["abs(N1-1)-abs(N1-3)"]<=spread)&
         (df["abs(N1-2)-abs(N1-3)"]<=spread)&
         (df["abs(O-1)"]<=spread)&
         (df["abs(O-2)"]<=spread)&
         (df["abs(O-3)"]<=spread)
         ]





#%%

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import pi

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




termsDict = [ 
    ("O-1", "G_{0, 0}"),
            ("O-2","G_{1, 1}"),
            ("O-3","G_{2, 2}"),
            ("N1-1","G_{0, 1}"),
            ("N1-2","G_{1, 2}"),
            ("N1-3","G_{0, 2}")]
            

a=35 
phi1s =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
phi2s =  [0, pi/7, pi/6, pi/5, pi/4, pi/3, pi/2]
centre1 = 1
centre2 = 2
apply = [np.abs, np.real, np.imag]
omegaMax = 20
omegaMin = 4
omegaMultiplier = 2
ymax = None
ymin = None
form = "Tri-RemoveGauge"#"Tri"
func1Name = "Cosine"##"Blip"#"RampHalf"#
func2Name = "Cosine"##"Blip"#"RampHalf"#


for look, matrixEl in termsDict:

    labels = [r"$|" +matrixEl+"|$", 
              r"$\mathrm{Real} \{"+matrixEl+r"\}$",
              r"$\mathrm{Imag} \{"+matrixEl+r"\}$"]
    
    sz =16
    
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(sz,sz/3),
                           constrained_layout=True, sharey=True)
    
    
    # df.loc[df['form'] == 'SS-p','phi offset'] = np.nan
            
    for omegaMultiplier in [2,3, 1.5]:
        for nc, phi1 in enumerate(phi1s):
            for nc2, phi2 in enumerate(phi2s):
    
                df_plot = dfn[(dfn['form']==form)&
                          (dfn['a1']==a)&
                          (dfn['a2']==a)&
                          (dfn['phi1']==phi1)&
                          (dfn['phi2']==phi2)&
                           (dfn["omega multiplier"]==omegaMultiplier)&
                          (dfn["centre1"]==centre1)
                          # (dfn["func1"]==func1Name)&
                          # (dfn["func2"]==func2Name)
                          ]
                    # df_plot.loc[:,"x-axis"] = df_plot.loc[:,"a"]/df_plot.loc[:,"omega"]
                    
            
                    
                if not df_plot.empty:
                    # df_plot["x-axis"] = df_plot.apply(
                    #     lambda row: row["a"]/row["omega"], axis=1)
                    
                    
                    # df_plot = df_plot.sort_values(by=['x-axis'])
                    
                    df_plot = df_plot.sort_values(by=['omega1'])
                    df_plot = df_plot[df_plot["omega1"] < omegaMax]
                    df_plot = df_plot[df_plot["omega1"] > omegaMin]
                    
                    for n1, f in enumerate(apply):
                        
                        ax[n1].plot(df_plot["omega1"], f(df_plot[look].values), 'x',
                                    label= (r'$\phi_1=' + PhiString(phi1) + r', \phi_2=' + PhiString(phi2) +r'\pi' 
                                            + ", A=" + str(a)
                                            + r", \omega_2 / \omega_1 = "+str(omegaMultiplier)+r'$')
                                    )
                        
                        # ax[n1].plot(df_plot["x-axis"], f(df_plot[look].values), 
                        #             label=
                        #                 r'$\phi_1=$'+str(round(phi/pi, 2))+r'$\pi$'
                        #                 +", A="+str(a))
                        
                        # ax[n1].set_xlabel(r'$A/\omega$')
                        ax[n1].set_xlabel(r'$\omega$')
                        ax[n1].set_title(labels[n1])
        #            ax[n1].set_ylim((-0.5, 0.5))
        
    for i in range(3):
        ax[i].axhline(y=0, color='0.9', linestyle='--')
    
    handles_legend, labels_legend = ax[1].get_legend_handles_labels()    
    fig.legend(handles_legend, labels_legend, loc='upper right')
    # plt.grid(True)
    ax[0].set_ylim([ymin, ymax])
    plt.show()
    
    #%%
    
def RemoveWannierGauge(matrix, c, N):
    phase = np.angle(matrix[c-1,c])
    phase = phase - np.pi #because it should be np.pi
    gaugeMatrix = np.identity(N, dtype=np.complex128)
    gaugeMatrix[c,c] = np.exp(-1j*phase)
    matrix = np.matmul(np.matmul(np.conj(gaugeMatrix), matrix), gaugeMatrix)
    return matrix
        



T = 2*pi/omega1
omega2 = 3*omega1
# elif form =="DS-p" or form == "SSDF-p":
#     omega2 = omegaMultiplier*omega1
#     aInput = [a1,a2]
#     omegaInput = [omega1,omega2]
#     phiInput = [phi, phi+phiOffset]

# calculate effective Hamiltonian 
paramss = [[a, omega1, phi1, onsite1], [a, omega2, phi2, onsite2]]
UT, HF = CreateHFGeneral(N, centres, funcs, paramss, T, circleBoundary)

for site in range(N):
    HF = RemoveWannierGauge(HF, site, N)
                    
    