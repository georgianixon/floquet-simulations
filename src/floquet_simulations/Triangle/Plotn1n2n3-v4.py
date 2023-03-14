# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:46:57 2022

@author: Georgia
"""

from scipy.special import jv
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from numpy import pi, exp, sin, cos
from math import gcd
import pandas as pd
place = "Georgia"
import matplotlib as mpl
# import seaborn as sns
# import sys
# sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
# from hamiltonians import CreateHFGeneral
# from hamiltonians import Cosine, ConvertComplex

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

def round2(i):
    return np.round(i,2)

def FloatToStringSave(a):
    return str(a).replace(".", "p")

def MaxDeltas2(ns, num_gaps = 4):
    '''Each of the maximally differing successive pairs
       in ns, each preceded by the value of the difference.
    '''
    pairs = list(enumerate(np.diff(ns)))
    pairso = max(pairs, key=lambda ab: ab[1])    
    first_gap = (pairso[1],( ns[pairso[0]], ns[pairso[0]+1]))
    pairs = np.delete(pairs, pairso[0], 0)
    pairso = max(pairs, key=lambda ab: ab[1])   
    second_gap = (pairso[1],( ns[int(pairso[0])], ns[int(pairso[0]+1)]))
    if num_gaps ==2:
        return first_gap, second_gap
    else:
        pairs = np.delete(pairs, pairso[0], 0)
        pairso = max(pairs, key=lambda ab: ab[1])
        third_gap =  (pairso[1],( ns[int(pairso[0])], ns[int(pairso[0]+1)]))
        if num_gaps == 3:
            return first_gap, second_gap, third_gap
        else:
            pairs = np.delete(pairs, pairso[0], 0)
            pairso = max(pairs, key=lambda ab: ab[1])
            fourth_gap =  (pairso[1],( ns[int(pairso[0])], ns[int(pairso[0]+1)]))
            return first_gap, second_gap, third_gap, fourth_gap
            

def SecondDelta(ns):
    '''Each of the maximally differing successive pairs
       in ns, each preceded by the value of the difference.
    '''
    pairs = [
        (abs(a - b), (a, b)) for a, b
        in zip(ns, ns[1:])
    ]
    delta = max(pairs, key=lambda ab: ab[0])
    d = dict(pairs)
    del d[delta[0]]
    second_max = max(d)
    return (second_max, d[second_max])

def MaxDelta(ns):
    '''Each of the maximally differing successive pairs
       in ns, each preceded by the value of the difference.'''
    pairs = [
        (abs(a - b), (a, b)) for a, b
        in zip(ns, ns[1:])
    ]
    delta = max(pairs, key=lambda ab: ab[0])
    return delta
    
dataLoc = "D:/Data/Set13-alpha=1,beta=2,omega=9/"
# dfO = pd.read_csv(dataLoc+"Summaries/FT-Min.csv",
                          # index_col=False)

dfO = pd.read_csv("D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/FT/FT-Min,phi3=0.csv", 
                  index_col=False)



#%%

"""
2) Plot showing values in lower triangle, non accumulative
"""

saveFig = ("C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs"+
           "/ShakingTriangle/Relative Hopping Triangle/V3/"+
           "alpha=1,beta=2,omega0=9,NonAccum,FT/")

saveFig = ("D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/FT/reltriangleplots"+
           "/phi3=0,mincovering2/")

color_var = "A3"
time_var = "A2"

if time_var=="A3":
    time_var_list = np.linspace(37, 38.5, 31 )
    #color var = A2 so
    color_var_min = 0
    color_var_max = 18.5
elif time_var == "A2":
    time_var_list = np.linspace(0, 18.5, 18*20+10+1)
    color_var_min = 37
    color_var_max = 38.5


alpha = 1; beta = 2; omega0=8; phi3=0
type_calc = "FT"
title_type = "First Term"

# for i, A3 in enumerate(np.linspace(0, 30.95, 620)):
for i, time_var_i in enumerate(time_var_list):
    time_var_i = np.round(time_var_i, 3)
  
    
    dfP = dfO[
        # (dfO.beta == beta)
             # &(dfO.alpha == alpha)
               # &(dfO.omega0 == omega0)
             (dfO[time_var] == time_var_i)
             &(dfO[color_var] <=color_var_max)
             &(dfO[color_var] >= color_var_min)
                # &(dfO["phi3/pi"]==phi3)
                      ]
    
    dfP = dfP.sort_values(by=[color_var])
    
    
    xLT = dfP[type_calc+"-LowerT.X"]
    yLT = dfP[type_calc+"-LowerT.Y"] 
    
    
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(xLT, yLT, s=3, c=dfP[color_var].to_numpy(), 
                    cmap="jet", marker=".")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel(r"$\frac{J_{\mathrm{med}}}{J_{\mathrm{max}}}$", 
                  fontsize = 24)
    ax.set_ylabel(r"$\frac{J_{\mathrm{min}}}{J_{\mathrm{max}}}$", 
                  rotation = 0, fontsize = 24,  labelpad = 20)
    cbar = plt.colorbar(sc)
    title = (title_type + r", $\alpha=" + str(alpha) + r", \beta="+str(beta)
             + r", \omega_0=" + str(omega0) 
             + r", \phi_3 "
              + r"= 0 "
             # + r" \in \{0, 2 \pi \}"             
             r",A_2="+f'{time_var_i:.2f}'
             # str(round(A3,2)) 
             + r"$")
    plt.suptitle(title)
    cbar.ax.set_ylabel(r"$A_3$", rotation=0, labelpad=10)
    plt.savefig(saveFig+"Frame"+str(i)+".png", format='png', bbox_inches='tight')
    plt.show()

#%%


"""
2) Plot showing values in lower triangle, non accumulative
"""

saveFig = ("C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs"+
           "/ShakingTriangle/Relative Hopping Triangle/V3/"+
           "alpha=1,beta=2,omega0=9,NonAccum,FT/")

saveFig = ("D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/FT/reltriangleplots"+
           "/phi3=0,mincovering2/")

color_var = "A2"
time_var = "A3"


alpha = 1; beta = 2; omega0=8; phi3=0
type_calc = "FT"
title_type = "First Term"

# for i, A3 in enumerate(np.linspace(0, 30.95, 620)):
for i, A3 in enumerate(np.linspace(37, 38.5, 31)):
    A3 = np.round(A3, 3)
  
    
    dfP = dfO[
        # (dfO.beta == beta)
             # &(dfO.alpha == alpha)
               # &(dfO.omega0 == omega0)
             (dfO.A3 == A3)
             &(dfO.A2 <=18.50)
                # &(dfO["phi3/pi"]==phi3)
                      ]
    
    dfP = dfP.sort_values(by=['A2'])
    
    
    xLT = dfP[type_calc+"-LowerT.X"]
    yLT = dfP[type_calc+"-LowerT.Y"] 
    
    
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(xLT, yLT, s=3, c=dfP.A2.to_numpy(), cmap="jet", marker=".")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel(r"$\frac{J_{\mathrm{med}}}{J_{\mathrm{max}}}$", fontsize = 24)
    ax.set_ylabel(r"$\frac{J_{\mathrm{min}}}{J_{\mathrm{max}}}$", rotation = 0, fontsize = 24,  labelpad = 20)
    cbar = plt.colorbar(sc)
    title = (title_type + r", $\alpha=" + str(alpha) + r", \beta="+str(beta)
             + r", \omega_0=" + str(omega0) 
             + r", \phi_3 "
              + r"= 0 "
             # + r" \in \{0, 2 \pi \}"             
             r",A_3="+f'{A3:.2f}'
             # str(round(A3,2)) 
             + r"$")
    plt.suptitle(title)
    cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
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

#%%

"""
Pick square in the lower triangle (brackets [ ])
See what sort of phases we can get in that square
"""

dataSave = 'D:/Data/Set13-alpha=1,beta=2,omega=9/HE/'

calc_type = "HE"
import random
import time
rad = 0.05


df = pd.DataFrame(columns = ["CentreX", "CentreY", "Radius", "nPoints",
                             "MaxDelta", "MaxPhaseOpening", "MaxPhaseClosing", 
                             "SecondMaxDelta", "SecondMaxPhaseOpening", "SecondMaxPhaseClosing"])
i = len(df)
for sqCentreX in np.linspace(0,1,101)[:-1]:
    for sqCentreY in np.linspace(0.0, sqCentreX, round((sqCentreX)/0.01 + 1)):

        
        sqCentreX = np.round(sqCentreX, 3)
        sqCentreY = np.round(sqCentreY, 3)
        print(sqCentreX, sqCentreY)
        
        dfP = dfO[((dfO[calc_type+"-LowerT.X"] - sqCentreX)**2 + (dfO[calc_type+"-LowerT.Y"] - sqCentreY)**2 <= rad**2)]
        
        phases = dfP[calc_type+"-Plaq-PHA"].to_numpy()
        n_points = len(phases)

        phases = np.sort(phases)
        maxDelta, secondMaxDelta = MaxDeltas2(phases)
        df.loc[i] = [sqCentreX, sqCentreY, rad, n_points, maxDelta[0], maxDelta[1][0], maxDelta[1][1], 
                     secondMaxDelta[0], secondMaxDelta[1][0], secondMaxDelta[1][1]]
        i +=1

st = time.time()    
df.to_csv(dataSave + "PhasesPlot.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")

dfP = df
#%% 

dataSave = 'D:/Data/Set13-alpha=1,beta=2,omega=9/HE/'
# dataLoc = 'D:/Data/Set12-alpha=1,beta=2,omega=8/'

dfP = pd.read_csv(dataSave  + "PhasesPlot.csv",
                          index_col=False)


type_calc = "Stroboscopic, Second Gap"
delta_type = "SecondMaxDelta"
gap_type_opening = "SecondMaxPhaseOpening"
gap_type_closing = "SecondMaxPhaseClosing"
save_string = "Second"


omega0 = 9
rad = 0.05

sz = 3
fig, ax = plt.subplots(figsize=(1.6*sz,sz))
sc = ax.scatter(dfP.CentreX, dfP.CentreY, c=dfP[delta_type], 
                norm = mpl.colors.Normalize(vmin=0, vmax=pi), s=1, cmap="jet", marker=".")
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlabel(r"$\frac{\mathrm{J}_{\mathrm{med}}}{\mathrm{J}_{\mathrm{max}}}$",  fontsize=20)
ax.set_ylabel(r"$\frac{\mathrm{J}_{\mathrm{min}}}{\mathrm{J}_{\mathrm{max}}}$", rotation = 0, labelpad=20, fontsize=20)
cbar = plt.colorbar(sc, ticks=[0, pi/8, pi/4, 3*pi/8,  pi/2, 5*pi/8, 3*pi/4, 7*pi/8, pi])
cbar.ax.set_ylabel(r"$\Delta(\phi)_{\mathrm{max}}$", rotation=0, labelpad=40)
cbar.ax.set_yticklabels(["0", 
                         r"",#r"$\pi/8$",
                         "$\pi/4$",
                          "",#"$3 \pi/8$",
                          r"$\pi/2$",
                          "",#r"$5\pi/8$",
                          r"$3\pi/4$",
                          "",#r"$7\pi/8$",
                          r"$\pi$" ])
ax.set_title(type_calc + r", Circle Radius = "+str(rad)+r", $\omega_0 ="+str(omega0)+r"$")
plt.savefig(dataSave + 'PhaseOpenings'+save_string+'Gap.png', dpi=300, format="png", bbox_inches='tight')
plt.show()  



sz = 3
fig, ax = plt.subplots(figsize=(1.6*sz,sz))
sc = ax.scatter(dfP.CentreX, dfP.CentreY, c=dfP.nPoints, s=1, 
                norm=mpl.colors.LogNorm(), cmap="jet", marker=".")
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlabel(r"$\frac{\mathrm{J}_{\mathrm{med}}}{\mathrm{J}_{\mathrm{max}}}$",  fontsize=20)
ax.set_ylabel(r"$\frac{\mathrm{J}_{\mathrm{min}}}{\mathrm{J}_{\mathrm{max}}}$", rotation = 0, labelpad=20, fontsize=20)
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel("$N_{\mathrm{points}}$", rotation=0, labelpad=25)
ax.set_title(type_calc + r", Circle Radius = "+str(rad)+r", $\omega_0 =" + str(omega0) + r"$")
plt.savefig(dataSave + 'NumOfPoints.png', dpi=300, format="png", bbox_inches='tight')
plt.show()  
        

   
fig, ax = plt.subplots(figsize=(1.6*sz,sz))
sc = ax.scatter(dfP[gap_type_opening], dfP[gap_type_closing], c=dfP[delta_type],
                norm = mpl.colors.Normalize(vmin=0, vmax=pi),
                s=1, cmap="jet", marker=".")
ax.set_xticks([-pi, -pi/2, 0,pi/2, pi])
ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_yticks([-pi, -pi/2, 0,pi/2, pi])
ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_xlabel(r"$\phi_{\mathrm{open}}$")
ax.set_ylabel(r"$\phi_{\mathrm{close}}$", rotation = 0, labelpad=10)
cbar = plt.colorbar(sc, ticks=[0, pi/8, pi/4, 3*pi/8,  pi/2, 5*pi/8, 3*pi/4, 7*pi/8, pi])
cbar.ax.set_ylabel(r"$\Delta(\phi)_{\mathrm{max}}$", rotation=0, labelpad=35)
cbar.ax.set_yticklabels(["0", 
                         r"",#r"$\pi/8$",
                         "$\pi/4$",
                          "",#"$3 \pi/8$",
                          r"$\pi/2$",
                          "",#r"$5\pi/8$",
                          r"$3\pi/4$",
                          "",#r"$7\pi/8$",
                          r"$\pi$" ])
ax.set_title(type_calc + r", Circle Radius = "+str(rad)+r", $\omega_0 =" + str(omega0) + r" $")
plt.savefig(dataSave + 'PhaseOpeningSizesOnRelTriangle'+save_string+'Gap.png', dpi=300, format="png",bbox_inches='tight')
plt.show()



        
        
        
        
        
        
        
        
        

