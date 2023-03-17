# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:46:57 2022

@author: Georgia
"""


import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, sin, cos
import pandas as pd
place = "Georgia"
import matplotlib as mpl
# import seaborn as sns
import sys
import time
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")

def Plot():
    # sns.set(style="darkgrid")
    # sns.set(rc={'axes.facecolor':'0.96'})
    size=18
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
    oxfordblue = "#061A40"
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    red = "#FC4445"
    newred = "#E4265C"
    flame = "#DD6031"
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                    CB91_Purple,
                    # CB91_Violet,
                    'dodgerblue',
                    'slategrey', newred]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
Plot()

def unique(a):
    unique, counts = np.unique(a, return_counts=True)
    return np.asarray((unique, counts)).T


def FloatToStringSave(a):
    return str(a).replace(".", "p")

import matplotlib as mpl

def make_segmented_cmap(): 
    white = '#e3e3e3'
    black = '#1c1c1c'
    lightred = "#ff80ee"
    darkred = "#b000a4"
    red = '#ff00d4'
    lightblue="#80eeff"
    darkblue="#00a4b0"
    blue = '#00d4ff'
    anglemap = mpl.colors.LinearSegmentedColormap.from_list(
        'anglemap', [black, darkred, red, lightred, white, 
                     lightblue,blue, darkblue,black], N=256, gamma=1)
    return anglemap
segmented_cmap = make_segmented_cmap()


def make_segmented_cmap(): 
    white = '#e3e3e3'
    black = '#3d3d3d'
    lightred = "#f9cfff"#ffa1dc"
    darkred = "#c72800"
    red = '#ff0091'
    lightblue="#cffff9"#"#a1dcff"
    darkblue="#2800c7"
    blue = '#0091ff'
    anglemap = mpl.colors.LinearSegmentedColormap.from_list(
        'anglemap', [black, darkred, red, lightred, white, 
                     lightblue,blue, darkblue,black], N=256, gamma=1)
    return anglemap
segmented_cmap = make_segmented_cmap()
dataLoc = "D:Data/Merges/alpha=1,beta=2,omega=8,0-40/FT/"

#%%
dfO = pd.read_csv(dataLoc+"FT-Min,phi3=0.csv",
                          index_col=False)
# dataLoc = "D:/Data/Merges/alpha=1,beta=2,omega=8/HE/"
# dfO = pd.read_csv(dataLoc+"HE-Min.csv",
#                           index_col=False)


# dfP = dfO[(dfO.A2 <2)&
#           (dfO.A3<2) ]
          
#%%

"""
2) Plot showing values in lower triangle, non accumulative
"""

# saveFig = ("C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs"+
#            "/ShakingTriangle/Relative Hopping Triangle/V3/"+
#            "alpha=1,beta=2,omega=8,granularphi,NonAccum,HE/")
# saveFig = ("D:/Data/Set12-alpha=1,beta=2,omega=8/HE/"
#             +"continuityplots/phi3=0p45/"
#             # +"alpha=1,beta=2,omega=8,phi3=0p15/"
#             )
# saveFig = ("D:/Data/Merges/alpha=1,beta=2,omega=8/HE/alpha=1,beta=2,omega=8,phi3=0,granularphi,NonAccum,HE/")
# saveFig = ("D:/Data/Set12-alpha=1,beta=2,omega=8/FT/fluxplots/phi3=0p3/")

alpha = 1; beta = 2; omega0=8; 

type_calc = "FT"
title_type = "First Term"#"Stroboscopic"
#%%

jmin = r"$J_{\mathrm{min}}$"
jmed = r"$J_{\mathrm{med}}$"
jmax = r"$J_{\mathrm{max}}$"

phi3s_full = np.linspace(0, 200, 201)
remove = set(np.linspace(0,195,14))
phi3s = [round(ii/100, 2) for ii in phi3s_full if ii in remove]

   

for phi3_frac, phi3save in zip([1.05, 1.2, 1.35, 1.5, 1.65, 1.8, 1.95], 
                               ["1p05", "1p2", "1p35", "1p5", "1p65", "1p8", "1p95"]):
    
    st = time.time()  
    df = dfO[np.round(dfO["phi3/pi"],2)==phi3_frac]
    df=df.sort_values(by=['A2', 'A3'])
    df = df.groupby(by=["A2", "A3", "omega0", "alpha", "beta", "phi3/pi"],
                      dropna=False).agg({
                            type_calc+'-Plaq-PHA':np.mean,
                            type_calc+'-LowerT.X':np.mean,
                            type_calc+'-LowerT.Y':np.mean,
                            }).reset_index()
      
    df.to_csv(dataLoc +  "FT-Min,phi3="+phi3save+".csv",
                      index=False, 
                      )
    et = time.time() 
    print("   process took", np.round(et - st, 1), "s")

#%%
dfs_lst = []
for phi3_frac, phi3save in zip(
        [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
                               # [ 1.05, 1.2, 1.35, 1.5, 1.65, 1.8, 1.95], 
                                ["0", "0p15", "0p3", "0p45", "0p6", "0p75",
                                "0p9"]):
                                # ["1p05", "1p2", "1p35", "1p5", "1p65", 
                                # "1p8", "1p95"]):
    df = pd.read_csv(dataLoc +  "FT-Min,phi3="+phi3save+".csv",
                      index_col=False
                      )
    dfs_lst.append(df)
    print(phi3_frac)
    

dfO = pd.concat(dfs_lst, ignore_index=True)
    #%%
# dfO["phi3/pi"] = np.round(dfO["phi3/pi"], 2)

    
# for phi3_frac, phi3save in zip([0, 0.3, 0.45, 0.6, 0.75, 0.9, 0.15], ["0", "0p3", "0p45", "0p6", "0p75", "0p9", "0p15"]):
# dfO = pd.read_csv(dataLoc+type_calc+"-Min,phi3="+phi3save+".csv",
#                       index_col=False)
saveFig=("D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/"+type_calc
         +"/fluxplots/phi3=0,0.15,..,0.9/")

# for i, A3 in enumerate(np.linspace(0, 30.95, 620)):
# for i, A3 in enumerate(np.linspace(0, 40.95, 41*20 )):
for i, A3 in enumerate(np.linspace(0, 40.9, 41*10 )):
# for i, A3 in enumerate(np.linspace(0, 30, 31)):
    A3 = np.round(A3, 2)
  
    
    dfP = dfO[
        # (dfO.beta == beta)
             # &(dfO.alpha == alpha)
               # &(dfO.omega0 == omega0)
             (dfO.A3 == A3)
                # &(dfO["phi3/pi"]==phi3_frac)
                # &(np.round(dfO["phi3/pi"],2)==phi3_frac)
                      ]
    
    dfP = dfP.sort_values(by=['A2'])
    
    
    xLT = dfP[type_calc+"-LowerT.X"]
    yLT = dfP[type_calc+"-LowerT.Y"] 
    
    
    fig, ax = plt.subplots(figsize=(6,5))
    # sc = ax.scatter(xLT, yLT, s=3, c=dfP.A2.to_numpy(), cmap="jet", marker=".")
    sc = ax.scatter(xLT, yLT, s=3, c=dfP[type_calc+"-Plaq-PHA"].to_numpy(), 
                      norm = mpl.colors.Normalize(vmin=-pi, vmax=pi),
                      cmap=segmented_cmap,
                      # "twilight_shifted", 
                      marker=".")
    # sc = ax.scatter(xLT, yLT, s=3, c=dfP.continuity.to_numpy(),
    #                 norm = mpl.colors.Normalize(vmin=0, vmax=5), cmap="jet", marker=".")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel(r"$\frac{J_{\mathrm{med}}}{J_{\mathrm{max}}}$", fontsize = 24)
    ax.set_ylabel(r"$\frac{J_{\mathrm{min}}}{J_{\mathrm{max}}}$", 
                  rotation = 0, fontsize = 24,  labelpad = 20)
    title = (title_type + r", $\alpha=" + str(alpha) + r", \beta="+str(beta)
             + r", \omega_0=" + str(omega0) 
             + r", \phi_3 "
                # + r"=" + str(phi3_frac) 
                # + r"\pi"
                # + r" \in \{0, \frac{1}{100} \pi,... 2 \pi \}"    
                + r" \in \{0, 0.15 \pi,... 0.9 \pi \}"
                # +r" \in \{1.05, 1.2 \pi,... 1.95 \pi \}"
             +r",A_3="+f'{A3:.2f}'
             # str(round(A3,2)) 
             + r"$")
    plt.suptitle(title, fontsize=14)
    
    # for A2 colourbar
    # cbar = plt.colorbar(sc)
    # cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)

    # for continuity colourbar
    # cbar = plt.colorbar(sc, ticks=[0, 1,2,3,4,5])
    # cbar.ax.set_yticklabels([jmin+"=J12,"+"\n"+jmax+"=J23",
    #                           jmin+"=J12,"+"\n"+jmax+"=J31",
    #                           jmin+"=J23,"+"\n"+jmax+"=J12",
    #                           jmin+"=J23,"+"\n"+jmax+"=J31",
    #                           jmin+"=J31,"+"\n"+jmax+"=J12",
    #                           jmin+"=J31,"+"\n"+jmax+"=J23",
    #                           ])
    # cbar.ax.set_ylabel("", rotation=0, labelpad=10)
    
    #for flux colourbar
    cbar = plt.colorbar(sc, ticks = [-pi,
                                     -3*pi/4,
                                     -pi/2,
                                     -pi/4,
                                     0,
                                     # pi/8,
                                     pi/4,
                                     # 3*pi/8,
                                     pi/2,
                                      # 5*pi/8,
                                      3*pi/4,
                                      # 7*pi/8,
                                      pi])
    cbar.ax.set_ylabel("Flux", rotation=0, labelpad=15)
    cbar.ax.set_yticklabels([r"$-\pi$",
                             r"$-3\pi/4$",
                             r"$-\pi/2$",
                             r"$-\pi/4$",
                             "0", 
                             # "",
                             r"$\pi/4$", 
                             # "",
                             r"$\pi/2$",
                             # "", 
                              r"$3\pi/4$",r"$\pi$"
                              ])
    
    plt.savefig(saveFig+"Frame"+str(i)+".png", format='png', bbox_inches='tight')
    plt.show()


#%%%


"""
Plot showing values in lower triangle accumulative - First Term
"""


saveFig = ("C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs"+
           "/ShakingTriangle/Relative Hopping Triangle/V3/"+
           "alpha=1,beta=2,omega0=9,phi3=0,NonAccum,FT/")

alpha = 1; beta = 2; omega0=9; phi=0

for i, A3max in enumerate(np.linspace(0,30.95,620)):
    # print(i)
    A3max =  np.round(A3max, 2)
    fig, ax = plt.subplots(figsize=(6,5))

    for A3 in np.linspace(0, A3max, i+1):
        
        # print("    ", A2)
        A3 =  np.round(A3, 2)
        dfP = dfO[
            # (dfO.beta == beta)
                          # &(dfO.alpha == alpha)
                          # &(dfO.omega0 == omega0)
                            (dfO.A3 == A3)
                            # &(dfO["phi3/pi"]==phi)
                          ]
        
        dfP = dfP.sort_values(by=['A3'])
        xLT = dfP["HE-LowerT.X"]
        yLT = dfP["HE-LowerT.Y"] 
        sc = ax.scatter(xLT, yLT, s=3, c=dfP.A2.to_numpy(), cmap="jet", marker=".")
        
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    cbar = plt.colorbar(sc)
    title = (r"Stroboscopic, $\alpha="+str(alpha)+r", \beta="+str(beta)+
              r", \omega_0="+str(omega0)+
              # r", \phi_3="+str(phi3)+r"\pi"
             r", A_3="+str(A3max)
             + r", \phi_3 \in \{0, 2 \pi\} "
             +r"$")
    plt.suptitle(title)
    cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
    plt.savefig(saveFig+"Frame"+str(i)+".png", format='png', bbox_inches='tight')
    plt.show()

        
        
        
        
        
        
        
        
        

