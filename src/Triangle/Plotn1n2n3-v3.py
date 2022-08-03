# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:57:17 2022

@author: Georgia Nixon
"""

from scipy.special import jv
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from numpy import pi, exp, sin, cos
from math import gcd
import pandas as pd
place = "Georgia Nixon"
import matplotlib as mpl
import seaborn as sns
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
from hamiltonians import CreateHFGeneral
from hamiltonians import Cosine, ConvertComplex

# dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
# figLoc =  "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Presentations - Me/20220504_OxfordConf/"
# latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Analytics/"


dataLoc =  "E://"

def Plot():
        
    sns.set(style="darkgrid")
    sns.set(rc={'axes.facecolor':'0.96'})
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
bigData = pd.read_csv(dataLoc+"bigData2.csv",
                          index_col=False)
#%%
#order by time shift

# this characterises the shift between site 3 and site2 in terms of T
bigData["tshift_frac"]=bigData["phi3rel/pi"]/2 + bigData["phi2/pi"]/2 - bigData["phi2/pi"] 



def unique(a):
    unique, counts = np.unique(a, return_counts=True)
    return np.asarray((unique, counts)).T
def round2(i):
    return np.round(i,2)


bigData["tshift_frac"] = bigData["tshift_frac"].apply(round2)
counts = unique(bigData["tshift_frac"].values)

#%%
# """
# Phases - keep phi constant
#     |
#     |
# \xi |
#     |
#     |____________
#           A_3
# - \phi is constant (phiFrac)
# - compare Ham Evolution and First Term
# - pick A2 = 30
# """

#fixed quantities
phi2Frac = 0.6
phi3rel = 0.6
A2 = 9


phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/phiI=0.6,A2=30/"

titletype = "First Term"#"Ham Evolution"#, "First Term"]
column = "FT-Plaq-PHA"#"HE-Plaq-PHA"# "FT-Plaq-PHA"]

# titletype = "Ham Evolution"
# column = "HE-Plaq-PHA"
# colours = ["#D30C7B", "#1BB18C"]


# for  title1, saveTit in zip( titles, folder):
    
for omega0 in np.linspace(4,20,17):
# for omega0 in [20]:

    dfP = bigData[(bigData["phi2/pi"]==phi2Frac)&
              (bigData["omega0"]==omega0)
               &(bigData["A2"]==A2)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    # title = title1 + r", $\alpha=1$, $\beta=2$, $\phi="+str(phiFrac)+r"\pi$, $A_2 = 30$, $\omega_0="+str(omega0)+r"$"
    title = titletype + r", $\alpha=1$, $\beta=2$, $\phi_2="+str(phi2Frac)+r"\pi$, $A_2 = "+str(int(A2))+"$, $\omega_0="+str(omega0)+r"$"
    

        
    data = dfP[column].to_numpy()
    # if column == "FT-J31-PHA":
    #     print(1)
    #     data = np.conj(data)

    x = dfP["A3"].to_numpy()
    # x = dfO["omega0"].to_numpy()
    sc = ax.scatter(x, data, s=3, c=dfP["tshift_frac"].to_numpy(), cmap="jet", marker=".")        # ax.plot(x, data, 'x', ms=1, )
    
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    # ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    # ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    # ax.set_xlabel(r"$\omega_0$")
    ax.set_xlabel(r"$A_3$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    cbar = plt.colorbar(sc)
    
    cbar.ax.set_ylabel(r"$\frac{t_{0}  }{T}$", rotation=0, labelpad=30)
    ax.set_title(title)
    # plt.savefig(phasesFigLoc+saveTit+"/"+saveTit+"Phases,alpha=1,beta=2,A2=30,phiI=0.6,omega0="+str(omega0)+".png", format='png', bbox_inches='tight')
    plt.show()
        
      
    
#%%

# """
# Phases - keep phi constant
#     |
#     |
# \xi |
#     |
#     |____________
#           A_3
# - \phi is constant (phiFrac)
# - pick Ham Evolve or First Term ()
# - have all A2 vals shown as colourbar
# """

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/phiI=0.6,Scatter/"

  # C:\Users\Georgia Nixon\OneDrive - University of Cambridge\MBQD\Figs\ShakingTriangle\Phases\V2\FirstTerm
title1 =  "Ham Evolution"
column = "HE-Plaq-PHA"
folder = "HamEvolve"


# title1 =  "First Term" 
# column = "FT-J31-PHA"
# columnTitle = "J31"
# folder = "FirstTerm" 


phi2Frac = 0.2
tshift_frac = 0.7


    
for omega0 in np.linspace(4,20,17):

    dfP = bigData[(bigData["phi2/pi"]==phi2Frac)&
              (bigData["omega0"]==omega0)
              &(bigData["tshift_frac"]==tshift_frac)
               # &(dfO["A2"]==30)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    title = title1 +r", $\alpha=1, \beta=2, \phi_2="+str(phi2Frac)+r"\pi, \omega_0="+str(omega0)+", 2 \frac{t_0}{T} = "+str(tshift_frac)+r"$"
    

    data = dfP[column].to_numpy()
    x = dfP["A3"].to_numpy()

    sc = ax.scatter(x, data, s=3, c=dfP.A2.to_numpy(), cmap="jet", marker=".")
        
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    # ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    # ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    # ax.set_xlabel(r"$\omega_0$")
    ax.set_xlabel(r"$A_3$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
    ax.set_title(title)
    # plt.savefig(phasesFigLoc+folder+"/"+folder+"Phases,"+columnTitle+",alpha=1,beta=2,phiI=0.6,omega0="+str(omega0)+".png", format='png', bbox_inches='tight')
    plt.show()



#%%

# """
# Phases - keep omega_0 constant (omega0=5)
#OLD ONE - USES REL PHASE NOT T SHIFT
#     |
#     |
# \xi |
#     |
#     |____________
#           \phi
# - omega_0 constant (omega0=5)
# - compare Ham Evolution and First Term
# - pick A2 = 30
# """


"""
Phases - vary phi
"""

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/omega0=10,A2=30/"


# lab = "Ham Evolution"#"First Term"
# column = "HE-Plaq-PHA"

lab = "First Term"
column = "FT-Plaq-PHA"



# co = "#D30C7B"#, "#1BB18C"]
omega0 = 5
A2 = 9

for A3 in np.linspace(0,30, 31):

    dfP = bigData[(bigData["omega0"]==omega0)&
              (bigData["A2"]==A2)&
               (bigData["A3"]==A3)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    # title = title1 + r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2=30$, $A_3="+str(A3)+r"$"
    title = lab+r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2="+str(A2)+"$, $A_3="+str(A3)+r"$"
     # lab, column, co in zip(titles, entry, colours):
    data = dfP[column].to_numpy()

    x = dfP["phi2/pi"].to_numpy()*pi
        # x = dfO["omega0"].to_numpy()

    sc = ax.scatter(x, data, s=3, c=dfP["phi3rel/pi"].to_numpy(), cmap="jet", marker=".")
    # ax.plot(x, data,'.', ms=1, color=co, label=lab)#"#D30C7B")
        
    ax.set_title(title)
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}+{2}$",  r"$2\pi$"])
    ax.set_xlabel(r"$\phi_2$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r"$\phi_3 - \phi_2$", rotation=0, labelpad=30)
    # plt.legend(loc="upper right")
    # plt.savefig(phasesFigLoc+saveTit+"/"+saveTit+"Phases,alpha=1,beta=2,A2="+str(A2)+",omega0="+str(omega0)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
    plt.show()
        


#%%

#NEW WITH T SHIFT
# """
# Phases - keep omega_0 constant (omega0=5)
#     |
#     |
# \xi |
#     |
#     |____________
#           \phi
# - omega_0 constant (omega0=5)
# - compare Ham Evolution and First Term
# - pick A2 = 30
# """


"""
Phases - vary phi
"""

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V3-relphase/A2=15,A3=15,omega0=5,ts/"


# lab = "Ham Evolution"#"First Term"
# column = "HE-Plaq-PHA"

lab = "First Term"
column = "FT-Plaq-PHA"



# co = "#D30C7B"#, "#1BB18C"]
omega0 = 5
A2 = 11
tshift_frac = 0.1

for tshift_frac in np.linspace(-1,1,41):
    tshift_frac = round2(tshift_frac)

    dfP = bigData[(bigData["omega0"]==omega0)&
              (bigData["A2"]==A2)&
               # (bigData["A3"]==A3)
               (bigData["tshift_frac"]==tshift_frac)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    # title = title1 + r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2=30$, $A_3="+str(A3)+r"$"
    title = lab+r", $\alpha=1, \beta=2, \omega_0="+str(omega0)+r", A_2="+str(A2)+r", \frac{2t_0}{T}="+str(tshift_frac)+r"$"
     # lab, column, co in zip(titles, entry, colours):
    data = dfP[column].to_numpy()

    x = dfP["phi2/pi"].to_numpy()*pi
        # x = dfO["omega0"].to_numpy()

    sc = ax.scatter(x, data, s=3, c=dfP["A3"].to_numpy(), cmap="jet", marker=".")
    # ax.plot(x, data,'.', ms=1, color=co, label=lab)#"#D30C7B")
        
    ax.set_title(title)
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    ax.set_xlabel(r"$\phi_2$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r"$A_3$", rotation=0, labelpad=30)
    # plt.legend(loc="upper right")
    # plt.savefig(phasesFigLoc+saveTit+"/"+saveTit+"Phases,alpha=1,beta=2,A2="+str(A2)+",omega0="+str(omega0)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
    plt.show()
        
    
#%%

#NEW WITH T SHIFT
# """
# Phases - keep omega_0 constant (omega0=5)
#     |
#     |
# \xi |
#     |
#     |____________
#           \phi
# - omega_0 constant (omega0=5)
# - compare Ham Evolution and First Term
# - pick A2 = 30
# """


"""
Phases - vary phi
"""

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V3-relphase/A2=15,A3=15,omega0=10,ts/"

def FloatToStringSave(a):
    return str(a).replace(".", "p")

omega0 = 10
A2 = 15
A3 = 15
# tshift_frac = 0.1

for ii, tshift_frac in enumerate(np.linspace(-1,1,41)):
    tshift_frac = round2(tshift_frac)

    dfP = bigData[(bigData["omega0"]==omega0)
              &(bigData["A2"]==A2)
               & (bigData["A3"]==A3)
               &(bigData["tshift_frac"]==tshift_frac)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    # title = title1 + r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2=30$, $A_3="+str(A3)+r"$"
    title = r"$\alpha=1, \beta=2, \omega_0="+str(omega0)+r", A_2="+str(A2)+r", A_3="+str(A3)+r", \frac{2t_s}{T}="+"{:.2f}".format(tshift_frac)+r"$"
     # lab, column, co in zip(titles, entry, colours):
    data = dfP[column].to_numpy()

    x = dfP["phi2/pi"].to_numpy()*pi
        # x = dfO["omega0"].to_numpy()

    # sc = ax.scatter(x, data, s=3, c=dfP["A3"].to_numpy(), cmap="jet", marker=".")
    ax.plot(x, dfP["FT-Plaq-PHA"].to_numpy(),'.', ms=4, color="#D30C7B", label="First Term")##D30C7B")
    ax.plot(x, dfP["HE-Plaq-PHA"].to_numpy(),'.', ms=4, color="#1BB18C", label="Ham Evolve")##D30C7B")

    
    ax.set_title(title)
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    ax.set_xlabel(r"$\phi_2$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    # cbar = plt.colorbar(sc)
    # cbar.ax.set_ylabel(r"$A_3$", rotation=0, labelpad=30)
    plt.legend(loc="upper right")
    plt.savefig(phasesFigLoc+"Phases,alpha=1,beta=2,A2="+str(A2)+",A3="+str(A3)+",omega0="+str(omega0)+",ts="+str(ii)+".png", format='png', bbox_inches='tight')
    plt.show()

#%%


"""
Phases - vary phi
"""

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/omega0=10,A2=30/"


# lab = "Ham Evolution"#"First Term"
# column = "HE-Plaq-PHA"

lab = "First Term"
column = "FT-Plaq-PHA"



# co = "#D30C7B"#, "#1BB18C"]
omega0 = 5
A2 = 11

for A3 in np.linspace(0,30, 31):

    dfP = bigData[(bigData["omega0"]==omega0)&
              (bigData["A2"]==A2)&
               (bigData["A3"]==A3)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    # title = title1 + r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2=30$, $A_3="+str(A3)+r"$"
    title = lab+r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2="+str(A2)+"$, $A_3="+str(A3)+r"$"
     # lab, column, co in zip(titles, entry, colours):
    data = dfP[column].to_numpy()

    x = dfP["phi2/pi"].to_numpy()*pi
        # x = dfO["omega0"].to_numpy()

    sc = ax.scatter(x, data, s=3, c=dfP["tshift_frac"].to_numpy(), cmap="jet", marker=".")
    # ax.plot(x, data,'.', ms=1, color=co, label=lab)#"#D30C7B")
        
    ax.set_title(title)
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    ax.set_xlabel(r"$\phi_2$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r"$\frac{2t_0}{T}$", rotation=0, labelpad=30)
    # plt.legend(loc="upper right")
    # plt.savefig(phasesFigLoc+saveTit+"/"+saveTit+"Phases,alpha=1,beta=2,A2="+str(A2)+",omega0="+str(omega0)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
    plt.show()
        


#%%



"""
Phases - vary phi
"""

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/omega0=10,A2=30/"


# lab = "Ham Evolution"#"First Term"
# column = "HE-Plaq-PHA"

lab = "First Term"
column = "FT-Plaq-PHA"



# co = "#D30C7B"#, "#1BB18C"]
omega0 = 5
A2 = 9
phi3rel_frac = 0.7

for A3 in np.linspace(0,30, 31):

    dfP = bigData[(bigData["omega0"]==omega0)&
              (bigData["phi3rel/pi"]==phi3rel_frac)&
               (bigData["A3"]==A3)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    # title = title1 + r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2=30$, $A_3="+str(A3)+r"$"
    title = lab+r", $\alpha=1, \beta=2, \omega_0="+str(omega0)+r", \phi_3 - \phi_2 ="+str(phi3rel_frac)+"\pi, A_3="+str(A3)+r"$"
     # lab, column, co in zip(titles, entry, colours):
    data = dfP[column].to_numpy()

    x = dfP["phi2/pi"].to_numpy()*pi
        # x = dfO["omega0"].to_numpy()

    sc = ax.scatter(x, data, s=3, c=dfP["A2"].to_numpy(), cmap="jet", marker=".")
    # ax.plot(x, data,'.', ms=1, color=co, label=lab)#"#D30C7B")
        
    ax.set_title(title)
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    ax.set_xlabel(r"$\phi_2$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=30)
    # plt.legend(loc="upper right")
    # plt.savefig(phasesFigLoc+saveTit+"/"+saveTit+"Phases,alpha=1,beta=2,A2="+str(A2)+",omega0="+str(omega0)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
    plt.show()
   


#%%

"""
Phases - vary phi
"""

phasesFigLoc = ("C:/Users/"+place+
                "/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases"+
                "/V3-relphase/A2=9,phi3rel=0p1/")


lab = "Ham Evolution"#"First Term"
column = "HE-Plaq-PHA"
saveTit = "HamEvolve"

lab = "First Term"s
column = "FT-Plaq-PHA"
saveTit = "FirstTerm"



# co = "#D30C7B"#, "#1BB18C"]
A2 = 9
phi3rel_frac = 0.1

for A3 in np.linspace(0,30, 31):

    dfP = bigData[(bigData["A2"]==A2)&
              (bigData["phi3rel/pi"]==phi3rel_frac)&
               (bigData["A3"]==A3)
               &(bigData["omega0"]<=10)
               &(bigData["omega0"]>=5)

              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    title = lab+r", $\alpha=1, \beta=2, A_2="+str(A2)+r", \phi_3 - \phi_2 ="+str(phi3rel_frac)+"\pi, A_3="+str(A3)+r"$"
    data = dfP[column].to_numpy()

    x = dfP["phi2/pi"].to_numpy()*pi
    sc = ax.scatter(x, data, s=3, c=dfP["omega0"].to_numpy(), cmap="jet", marker=".")
        
    ax.set_title(title)
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    ax.set_xlabel(r"$\phi_2$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r"$\omega_0$", rotation=0, labelpad=30)
    plt.savefig(phasesFigLoc+saveTit+"/"+saveTit+"alpha=1,beta=2,A2="+str(A2)+",A3="+str(A3)+",phi3relopi="+str(phi3rel_frac)+".png", format='png', bbox_inches='tight')
    plt.show()     
#%%

# """
# ABS VAL - keep omega_0 constant (omega0=10)
#     |
#     |
# \xi |
#     |
#     |____________
#           \phi
# - omega_0 constant (omega0=5)
# - compare Ham Evolution and First Term
# - color is A2
# - increase A3 over time
# """


"""
Phases - vary phi
"""
omega0 = 5

phasesFigLoc = ("C:/Users/"+place
                +"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/omega0="
                +str(omega0)+",Scatter/")

for ele in [ r"12",r"23", r"31"]:
    
    title1 =  "Ham Evolution"
    column = "HE-J"+ele+"-ABS"
    folder = "J"+ele+"AbsHamEvolve"
    ylabel=r"$|J_{"+ele+"}|$"
    
    # title1 =  "First Term" 
    # column = "FT-J"+ele+"-ABS"
    # folder = "J"+ele+"AbsFirstTerm" 
    # ylabel=r"$|J_{"+ele+"}|$"
    
    
        
    
    for A3 in np.linspace(0,30, 31):
    
        dfP = dfO[(dfO["omega0"]==omega0)&
                   (dfO["A3"]==A3)
                  ]
        
        fig, ax = plt.subplots(figsize=(5,3))
        title = title1 +r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_3="+str(A3)+r"$"
        
    
        data = dfP[column].to_numpy()
        x = dfP["phi3/pi"].to_numpy()*pi
        colour = dfP.A2.to_numpy()
        sc = ax.scatter(x, data, s=3, c=colour, cmap="jet", marker=".")
    
    
        ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
        ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
        ax.set_xlabel(r"$\phi$")
        
        ax.set_ylabel(ylabel, rotation=0, labelpad=12)
        ax.set_ylim([-0.1,1.1])
        
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
        # cbar.ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r"$3\pi/2$", r"$2\pi$"])
        
        ax.set_title(title)
        # plt.savefig(phasesFigLoc+folder+"/"+folder+",alpha=1,beta=2,omega0="+str(omega0)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
        plt.show()
    

    
#%%

# """
# Scatter plot
# Phases - keep omega0 constant
#     |
#     |
# \xi |
#     |
#     |____________
#           \phi
# - omega_0 is constant omega0=5
# - pick Ham Evolve or First Term ()
# - have all A2 vals shown as colourbar
# """

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/omega0=9,Scatter/"

# title1 =  "Ham Evolution"
# column = "HE-J31-PHA"
# folder = "HamEvolve"


title1 =  "First Term" 
column = "FT-J31-PHA"
folder = "FirstTerm" 


omega0 = 9
    
for A3 in np.linspace(0,30, 31):
    

    dfP = dfO[(dfO["omega0"]==omega0)&
               (dfO["A3"]==A3)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    title = title1 +r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_3="+str(A3)+r"$"
    

    data = dfP[column].to_numpy()
    x = dfP["phi3/pi"].to_numpy()*pi

    sc = ax.scatter(x, data, s=3, c=dfP.A2.to_numpy(), cmap="jet", marker=".")
        
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    ax.set_xlabel(r"$\phi$")
    # ax.set_xlabel(r"$A_3$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
    ax.set_title(title)
    # plt.savefig(phasesFigLoc+folder+"/"+folder+"Phases,alpha=1,beta=2,omega0="+str(omega0)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
    plt.show()



    
#%%

# """
# Scatter plot
# Phases - keep A2 constant
#     |
#     |
# \xi |
#     |
#     |____________
#        \omega_0
# - A2 is constant A2=25
# - pick Ham Evolve or First Term ()
# - have all \phi3 vals shown as colourbar
# - have A3 evolve over time from 0 - 30
# """

# title1 =  "Ham Evolution"
# column = "HE-J31-PHA"
# folder = "HamEvolve"


# title1 =  "First Term" 
# column = "FT-J31-ABS"
# folder = "FirstTerm" 


A2 = 30

phasesFigLoc = ("C:/Users/"+place
                +"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/A2="
                +str(A2)+",Scatter/")

for ele in [ r"12",r"23", r"31"]:
    
    # title1 =  "Ham Evolution"
    # column = "HE-J"+ele+"-ABS"
    # folder = "J"+ele+"AbsHamEvolve"
    # ylabel=r"$|J_{"+ele+"}|$"
    
    title1 =  "First Term" 
    column = "FT-J"+ele+"-ABS"
    folder = "J"+ele+"AbsFirstTerm" 
    ylabel=r"$|J_{"+ele+"}|$"
    
    

    for A3 in np.linspace(0,30, 31):
        
    
        dfP = dfO[(dfO["A2"]==A2)&
                   (dfO["A3"]==A3)
                  ]
        
        fig, ax = plt.subplots(figsize=(5,3))
        title = title1 +r", $\alpha=1$, $\beta=2$, $A_2="+str(A2)+r"$, $A_3="+str(A3)+r"$"
        
    
        data = dfP[column].to_numpy()
        x = dfP["omega0"].to_numpy()
        colour = dfP["phi3/pi"].to_numpy()*pi
        sc = ax.scatter(x, data, s=3, c=colour, cmap="jet", marker=".")
            
        # ax.set_ylabel(r"$\xi$", rotation=0)
        # ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
        # ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
        # ax.set_ylim([-pi-0.1, pi+0.1])
        
        ax.set_ylabel(ylabel, rotation=0, labelpad=10)
        ax.set_ylim([-0.1,1.1])
        # ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
        # ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
        
        ax.set_xticks([4,8,12,16,20])
        # ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
        ax.set_xlabel(r"$\omega_0$")
        
        cbar = plt.colorbar(sc, ticks=[0, pi/2, pi, 3*pi/2, 2*pi])
        cbar.ax.set_ylabel(r"$\phi_3$", rotation=0, labelpad=5)
        cbar.ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r"$3\pi/2$", r"$2\pi$"])
        ax.set_title(title)
        # plt.savefig(phasesFigLoc+folder+"/"+folder+",alpha=1,beta=2,A2="+str(A2)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
        plt.show()

#%%


# """
# Scatter plot
# ABS VALS - keep A2 constant
#     |
#     |
# \xi |
#     |
#     |____________
#        \omega_0
# - A2 is constant A2=30
# - pick Ham Evolve or First Term ()
# - have all \phi3 vals shown as colourbar
# - have A3 evolve over time from 0 - 30
# """

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/A2=30,Scatter/"

# title1 =  "Ham Evolution"
# column = "HE-J31-ABS"
# folder = "J31AbsHamEvolve"
# ylabel=r"$|J_{31}|$"

title1 =  "First Term" 
column = "FT-J31-ABS"
folder = "J31AbsFirstTerm" 
ylabel=r"$|J_{31}|$"


A2 = 25


for A3 in np.linspace(0,30, 31):
    

    dfP = dfO[(dfO["A2"]==A2)&
               (dfO["A3"]==A3)
              ]
    
    fig, ax = plt.subplots(figsize=(5,3))
    title = title1 +r", $\alpha=1$, $\beta=2$, $A_2="+str(A2)+r"$, $A_3="+str(A3)+r"$"
    

    data = dfP[column].to_numpy()
    x = dfP["omega0"].to_numpy()
    colour = dfP["phi3/pi"].to_numpy()*pi
    sc = ax.scatter(x, data, s=3, c=colour, cmap="jet", marker=".")
        
    # ax.set_ylabel(r"$\xi$", rotation=0)
    # ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    # ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    # ax.set_ylim([-pi-0.1, pi+0.1])
    
    ax.set_ylabel(ylabel, rotation=0, labelpad=12)
    ax.set_ylim([-0.1,1.1])
    # ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    # ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    
    ax.set_xticks([4,8,12,16,20])
    # ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    ax.set_xlabel(r"$\omega_0$")
    
    cbar = plt.colorbar(sc, ticks=[0, pi/2, pi, 3*pi/2, 2*pi])
    cbar.ax.set_ylabel(r"$\phi_3$", rotation=0, labelpad=5)
    cbar.ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r"$3\pi/2$", r"$2\pi$"])
    ax.set_title(title)
    # plt.savefig(phasesFigLoc+folder+"/"+folder+",alpha=1,beta=2,A2="+str(A2)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
    plt.show()



#%%


"""
ABSOLUTE VALUE RATIO PLOTS

1) Plot everything 
"""
omegaMin = 6
omegaMax = 20
A2Min = 25
A2Max = 30
A3Min = 0
A3Max = 30

sz =10

ms = 1.5
fig, ax = plt.subplots(figsize=(6,6))

for alpha in [1]:
    for beta in [2]:
        

        realOmegaMin = alpha*omegaMin
        realOmegaMax = alpha*omegaMax
        
        dfP = dfO[(dfO.beta == beta)
                  &(dfO.alpha == alpha)
                  # &(dfO.omega0 <= omegaMax)
                   &(dfO.omega0 >= omegaMin)
                  # &(dfO.A2 >= A2Min)
                  # &(dfO.A2 <= A2Max)
                  # &(dfO.A3 >= A3Min)
                  # &(dfO.A3 <= A3Max)
                  # &(dfO["LowerT.X"]>0.92)
                  # &(dfO["LowerT.Y"]>0.42)
                  # &(dfO["LowerT.Y"]<0.5)
                  ]
        
        if not dfP.empty:
            
            xLT = dfP["HE-LowerT.X"]
            yLT = dfP["HE-LowerT.Y"] 

            ax.plot(np.abs(xLT), np.abs(yLT), '.', label=r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r"$", markersize=ms)

            # ax.set_ylabel(r"$J_{31}/J_{23}$", rotation=0, labelpad=10)
            # ax.set_xlabel(r"$J_{12}/J_{23}$")
            ax.set_ylim([0,1])
            ax.set_xlim([0,1])

fig.suptitle(r"$\omega \in ["+str(omegaMin)+r", "+str(omegaMax)+r"], \> A_2 \in ["+str(A2Min)+r", "+str(A2Max)+r"],\>  A_3 \in ["+str(A3Min)+r", "+str(A3Max)+r"]$")
plt.legend(loc="upper right")
# fig.savefig(latexLoc+'Fig-n1n2n3.png', format='png', bbox_inches='tight')
plt.show()


#%%

# """
# plot showing how alpha and beta
# """

# fig, ax = plt.subplots(figsize=(6,6))
# alphas = np.linspace(0,1,1000)
# betas = np.linspace(0,1,1000)

# for i, alpha in enumerate(alphas):
#     for beta in betas[:i]:
#         if i == 0:
#             beta = betas[0]
#         lst = [alpha, beta, 1]  
#         ax.plot(alpha, beta, '.', color="#E4265C", markersize=5)
#         ax.plot(beta, alpha, '.', color="#E4265C", markersize=5)
#         if alpha!=0:
#             ax.plot(1/alpha, beta/alpha, '.', color='#47DBCD', markersize=5)
#             ax.plot( beta/alpha, 1/alpha,'.', color='#47DBCD', markersize=5)
#         if beta !=0:
#             ax.plot(alpha/beta, 1/beta, '.', color='darkblue', markersize=5)
#             ax.plot(1/beta, alpha/beta, '.', color='darkblue', markersize=5)
# ax.set_xlim([0,20])
# ax.set_ylim([0,20])
# plt.show()



#%%

"""
2) Plot showing values in lower triangle, non accumulative
"""
# dfN = dfO[(dfO["alpha"]==1)&(dfO["beta"]==2)&(dfO["omega"])]


saveFig = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/"
# for A2 in np.linspace(0,30,31):
alpha = 1; beta = 2; omega0=5; phi3=0
for A3 in np.linspace(0,30,31):
    A3 = np.round(A3, 1)
    
    # omegaMax= 20; omegaMin = 0;
    # A2 = 19
    # A2Min = 0; A2Max = 30; A3Min = 0; A3Max = 30
    
    dfP = dfO[(dfO.beta == beta)
             &(dfO.alpha == alpha)
              &(dfO.omega0 == omega0)
             &(dfO.A3 == A3)
              # &(dfO["phi3/pi"]==phi3)
                      ]
    
    dfP = dfP.sort_values(by=['A2'])
    
    
    xLT = dfP["FT-LowerT.X"]
    yLT = dfP["FT-LowerT.Y"] 
    
    
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(xLT, yLT, s=3, c=dfP.A2.to_numpy(), cmap="jet", marker=".")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    cbar = plt.colorbar(sc)
    title = (r"$\alpha="+str(alpha)+r", \beta="+str(beta)+
              r", \omega_0="+str(omega0)+
              # r", \phi_3="+str(phi3)+r"\pi"
             r", A_3="+str(A3)
             +r"$")
    plt.suptitle(title)
    cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
    # plt.savefig(saveFig+"alpha=1,beta=2,omega0=10,More,Accumulate/"+"Frame"+str(A2)+".png", format='png', bbox_inches='tight')
    plt.show()


#%%%


"""
Plot showing values in lower triangle accumulative - First Term
"""

saveFig = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs2/"


alpha = 1; beta = 2; omega0=5; phi=0
# omegaMax= 20; omegaMin = 0;

# A2 = 19
# A2Min = 0; A2Max = 30; A3Min = 0; A3Max = 30
# for A2 in np.linspace(0,30,31):
    
for i, A3max in enumerate(np.linspace(0,30,31)):#np.linspace(0,30,301)):
    # print(i)
    A3max =  np.round(A3max, 2)
    fig, ax = plt.subplots(figsize=(6,5))
    
    
    
    for A3 in np.linspace(0, A3max, i+1):
        
        # print("    ", A2)
        A3 =  np.round(A3, 2)
        
        dfP = dfO[(dfO.beta == beta)
                          &(dfO.alpha == alpha)
                          &(dfO.omega0 == omega0)
                            &(dfO.A3 == A3)
                            # &(dfO["phi3/pi"]==phi)
                          ]
        
        dfP = dfP.sort_values(by=['A3'])
        
        
        xLT = dfP["FT-LowerT.X"]
        yLT = dfP["FT-LowerT.Y"] 
        
        
        
        sc = ax.scatter(xLT, yLT, s=3, c=dfP.A2.to_numpy(), cmap="jet", marker=".")
        
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    cbar = plt.colorbar(sc)
    title = (r"$\alpha="+str(alpha)+r", \beta="+str(beta)+
              r", \omega_0="+str(omega0)+
              # r", \phi_3="+str(phi3)+r"\pi"
             r", A_3="+str(A3max)
             +r"$")
    plt.suptitle(title)
    cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
    # plt.savefig(saveFig+"alpha=1,beta=2,omega0=10,More,AccumulateR/"+"Frame"+str(A2max)+".png", format='png', bbox_inches='tight')
    plt.show()

#%%

"""
Pick square in the lower triangle (brackets [ ])
See what sort of phases we can get in that square
"""


# def maxDeltas(ns):
#     '''Each of the maximally differing successive pairs
#        in ns, each preceded by the value of the difference.
#     '''
#     pairs = [
#         abs(ns[i] - ns[i + 1]) for i
#         in range(len(ns)-1)]
#     delta = max(pairs)
#     return delta


def maxDeltas(ns):
    '''Each of the maximally differing successive pairs
       in ns, each preceded by the value of the difference.
    '''
    pairs = [
        (abs(a - b), (a, b)) for a, b
        in zip(ns, ns[1:])
    ]
    delta = max(pairs, key=lambda ab: ab[0])
 
    return delta

import random
rad = 0.05

# df = pd.DataFrame(columns = ["CentreX", "CentreY", "Radius", "MaxDelta", "PhaseOpening", "PhaseClosing"])
i = len(df)
for sqCentreX in np.linspace(0,1,101)[:-1]:
    for sqCentreY in np.linspace(0.0, sqCentreX, round((sqCentreX)/0.01 + 1)):

        
        sqCentreX = np.round(sqCentreX, 3)
        sqCentreY = np.round(sqCentreY, 3)
        
        # sqCentreX = 0.94
        # sqCentreY = 0.94
        # rad = 0.01
        print(sqCentreX, sqCentreY)
        
        dfP = dfO[((dfO["HE-LowerT.X"] - sqCentreX)**2 + (dfO["HE-LowerT.Y"] - sqCentreY)**2 <= rad**2)]
        
        # dfP = dfO[(dfO["HE-LowerT.X"] < sqCentreX+rad)&
        #           (dfO["HE-LowerT.X"] > sqCentreX-rad)&
        #           (dfO["HE-LowerT.Y"] < sqCentreY+rad)&
        #           (dfO["HE-LowerT.Y"] > sqCentreY-rad)
        #           ]
        
        phases = dfP["HE-J31-PHA"].to_numpy()
        
        # colour = dfP.A2.to_numpy()
        # yaxis = dfP["phi3/pi"].to_numpy()
        # title = (r"$\alpha=1$, $\beta=2$, centre$=("+"{0:.4g}".format(sqCentreX)+r","+"{0:4g}".format(sqCentreY)+r")$, rad$="+"{0:.4g}".format(rad)+r"$")
        # fig, ax = plt.subplots(figsize=(5,3))
        # sc = ax.scatter( phases,yaxis, s=1, c=colour, cmap="jet", marker=".")
        # ax.set_xticks([-pi, -pi/2, 0,pi/2, pi])
        # ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
        # ax.set_xlabel(r"effective $\phi$")
        # cbar = plt.colorbar(sc)
        # cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
        # ax.set_title(title)
        # ax.set_ylabel(r"$\frac{\theta_3}{\pi} $", rotation=0, labelpad = 12)
        # plt.show()
        
        
        # fig, ax = plt.subplots(figsize=(5,3))
        # sc = ax.scatter( phases,[0]*len(phases), s=1, c=colour, cmap="jet", marker=".")
        # ax.set_xticks([-pi, -pi/2, 0,pi/2, pi])
        # ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
        # ax.set_xlabel(r"effective $\phi$")
        # ax.set_yticks([0])
        # cbar = plt.colorbar(sc)
        # cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
        # ax.set_title(title)
        # plt.show()

        phases = np.sort(phases)
        maxDelta = maxDeltas(phases)
        df.loc[i] = [sqCentreX, sqCentreY, rad, maxDelta[0], maxDelta[1][0], maxDelta[1][1]]
        i +=1
        

#%%
rad = 0.05
dfP =df[(df.Radius == rad)
        &(df.CentreX !=1)
        &(df.CentreY != 0)
        ]
size=12
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
sz = 3
fig, ax = plt.subplots(figsize=(1.6*sz,sz))
sc = ax.scatter(dfP.CentreX, dfP.CentreY, c=dfP.MaxDelta, s=1, cmap="jet", marker=".")
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlabel(r"$\frac{\mathrm{J}_a}{\mathrm{J}_c}$",  fontsize=14)
ax.set_ylabel(r"$\frac{\mathrm{J}_b}{\mathrm{J}_c}$", rotation = 0, labelpad=10, fontsize=14)
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r"$\Delta(\phi)_{\mathrm{max}}$", rotation=0, labelpad=25)
ax.set_title("Circle Radius = "+str(rad))
plt.savefig(figLoc + 'PhaseOpenings.png', dpi=300, format="png", bbox_inches='tight')
plt.show()  
        
fig, ax = plt.subplots(figsize=(1.6*sz,sz))
sc = ax.scatter(dfP.PhaseOpening, dfP.PhaseClosing, c=dfP.MaxDelta, s=1, cmap="jet", marker=".")
ax.set_xticks([-pi, -pi/2, 0,pi/2, pi])
ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_yticks([-pi, -pi/2, 0,pi/2, pi])
ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_xlabel(r"$\phi_{\mathrm{open}}$")
ax.set_ylabel(r"$\phi_{\mathrm{close}}$", rotation = 0, labelpad=10)
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r"$\Delta(\phi)_{\mathrm{max}}$", rotation=0, labelpad=25)
ax.set_title("Circle Radius = "+str(rad))
plt.savefig(figLoc + 'PhaseOpeningSizesOnRelTriangle.png', dpi=300, format="png",bbox_inches='tight')
plt.show()


# sqCentreX = 0.5
# sqCentreY = 0.2
# rad = 0.01

# x = dfP["HE-LowerT.X"]
# y = dfP["HE-LowerT.Y"]
                                    
# fig, ax = plt.subplots(figsize=(5,3))
# sc = ax.scatter( x,y, s=1, cmap="jet", marker=".")
# plt.show()        
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

