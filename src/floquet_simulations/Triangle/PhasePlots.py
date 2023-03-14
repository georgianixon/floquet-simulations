# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:31:20 2022

@author: Georgia
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, sin, cos
import pandas as pd
place = "Georgia"
import matplotlib as mpl
import seaborn as sns
import sys
import time
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")

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

def MaxDeltas(ns, num_gaps = 4):
    '''Each of the maximally differing successive pairs
       in ns, each preceded by the value of the difference.
    '''
    pairs = [
        (i, abs(a - b), a, b) for i, (a, b)
        in enumerate(zip(ns, ns[1:]))
    ]
    pairs_max = max(pairs, key=lambda ab: ab[1])  
    first_gap = (pairs_max[1], (pairs_max[2], pairs_max[3]))
    pairs = np.delete(pairs, pairs_max[0], 0)
    pairs[:,0] = range(len(pairs))
    
    pairs_max = max(pairs, key=lambda ab: ab[1])  
    second_gap = (pairs_max[1], (pairs_max[2], pairs_max[3]))
    # if num_gaps ==2:
    #     return first_gap, second_gap
    # else:
    pairs = np.delete(pairs, int(pairs_max[0]), 0)
    pairs[:,0] = range(len(pairs))
    pairs_max = max(pairs, key=lambda ab: ab[1])

    third_gap =  (pairs_max[1], (pairs_max[2], pairs_max[3]))
        # if num_gaps == 3:
        #     return first_gap, second_gap, third_gap
        # else:
    pairs = np.delete(pairs, int(pairs_max[0]), 0)
    pairs[:,0] = range(len(pairs))
    pairs_max = max(pairs, key=lambda ab: ab[1])
    fourth_gap =  (pairs_max[1], (pairs_max[2], pairs_max[3]))
    return first_gap, second_gap, third_gap, fourth_gap

            

# def SecondDelta(ns):
#     '''Each of the maximally differing successive pairs
#        in ns, each preceded by the value of the difference.
#     '''
    # pairs = [
    #     (abs(a - b), (a, b)) for a, b
    #     in zip(ns, ns[1:])
    # ]
#     delta = max(pairs, key=lambda ab: ab[0])
#     d = dict(pairs)
#     del d[delta[0]]
#     second_max = max(d)
#     return (second_max, d[second_max])

# def MaxDelta(ns):
#     '''Each of the maximally differing successive pairs
#        in ns, each preceded by the value of the difference.'''
#     pairs = [
#         (abs(a - b), (a, b)) for a, b
#         in zip(ns, ns[1:])
#     ]
#     delta = max(pairs, key=lambda ab: ab[0])
#     return delta






#%%

"""
Pick square in the lower triangle (brackets [ ])
See what sort of phases we can get in that square
"""

# D:\Data\Merges\alpha=1,beta=2,omega=8,0-40\FT
dataLoc = "D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/FT/"
dfO = pd.read_csv(dataLoc+"FT-Min.csv",
                          index_col=False)


calc_type = "FT"

rad = 0.05


df = pd.DataFrame(columns = ["DataType", 
                             "CentreX", "CentreY", "Radius", "nPoints",
                             "MaxDelta", "MaxPhaseOpening", "MaxPhaseClosing", 
                             "SecondMaxDelta", "SecondMaxPhaseOpening", "SecondMaxPhaseClosing",
                             "ThirdMaxDelta", "ThirdMaxPhaseOpening", "ThirdMaxPhaseClosing",
                             "FourthMaxDelta", "FourthMaxPhaseOpening", "FourthMaxPhaseClosing"
                             ])
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
        maxDelta, secondMaxDelta, thirdMaxDelta, fourthMaxDelta = MaxDeltas(phases)
        df.loc[i] = [calc_type,
                     sqCentreX, sqCentreY, rad, n_points, 
                     maxDelta[0], maxDelta[1][0], maxDelta[1][1], 
                     secondMaxDelta[0], secondMaxDelta[1][0], secondMaxDelta[1][1],
                     thirdMaxDelta[0], thirdMaxDelta[1][0], thirdMaxDelta[1][1], 
                     fourthMaxDelta[0], fourthMaxDelta[1][0], fourthMaxDelta[1][1], 
                     ]
        i +=1

st = time.time()    
df.to_csv(dataLoc + "PhasesPlot.csv", index=False)
et = time.time()
print("   save took", np.round(et - st, 1), "s")


#%% 

dataSave = "D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/FT/"
# # dataLoc = 'D:/Data/Set12-alpha=1,beta=2,omega=8/'

# dfP = pd.read_csv(dataSave  + "PhasesPlot.csv",
                          # index_col=False)


calc_type = "First Term"#"Stroboscopic"#"First Term"

type_calc = calc_type + ", First Gap"
delta_type = "MaxDelta"
gap_type_opening = "MaxPhaseOpening"
gap_type_closing = "MaxPhaseClosing"
save_string = "First"


# type_calc = calc_type + ", Second Gap"
# delta_type = "SecondMaxDelta"
# gap_type_opening = "SecondMaxPhaseOpening"
# gap_type_closing = "SecondMaxPhaseClosing"
# save_string = "Second"




type_calc = calc_type + ", Third Gap"
delta_type = "ThirdMaxDelta"
gap_type_opening = "ThirdMaxPhaseOpening"
gap_type_closing = "ThirdMaxPhaseClosing"
save_string = "Third"


type_calc = calc_type + ", Fourth Gap"
delta_type = "FourthMaxDelta"
gap_type_opening = "FourthMaxPhaseOpening"
gap_type_closing = "FourthMaxPhaseClosing"
save_string = "Fourth"

omega0 = 8
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
ax.set_title(type_calc + r", Circle Radius = "+str(rad)
             +r", $\omega_0 ="+str(omega0)+r", A \in \{0,40\}"+r"$",
             fontsize=12)
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
ax.set_title(calc_type + r", Circle Radius = "+str(rad)
             +r", $\omega_0 =" + str(omega0)+r", A \in \{0,40\}"+r"$",
             fontsize=12)
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
ax.set_title(type_calc + r", Circle Radius = "+str(rad)
             +r", $\omega_0 =" + str(omega0)+r", A \in \{0,40\}"+r"$",
             fontsize=12)
plt.savefig(dataSave + 'PhaseOpeningSizesOnRelTriangle'+save_string+'Gap.png', dpi=300, format="png",bbox_inches='tight')
plt.show()


