# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:53:42 2022

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
import seaborn as sns
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
from hamiltonians import CreateHFGeneral
from hamiltonians import Cosine

dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/OldStuff/"
dfname = "TriangleRatios-v2.csv"


def ListRatiosInLowerTriangle(lst1a,lst1b, lst2a,lst2b, lst3a,lst3b):
    """
    Go thourgh (x1,y1), (x2,y2) (x3,y3) combinations and find the one in the bottom right triangle
    """
    N = len(lst1a)
    lowerTriListA = np.zeros(N)
    lowerTriListB = np.zeros(N)
    
    upperTriListX = np.zeros(N)
    upperTriListY = np.zeros(N)
    
    # counts = np.zeros(N)
    
    for i, (a1, b1, a2, b2, a3, b3) in enumerate(list(zip(lst1a, lst1b, lst2a, lst2b, lst3a, lst3b))):
        # count = 0
        if a1 <=1 and b1 <=1:
            # count +=1
            if b1<=a1:
                lowerTriListA[i] = a1
                lowerTriListB[i] = b1
                
                upperTriListX[i] = b1
                upperTriListY[i] = a1
            else:
                lowerTriListA[i] = b1
                lowerTriListB[i] = a1
                
                upperTriListX[i] = a1
                upperTriListY[i] = b1
        elif a2 <=1 and b2 <=1:
            # count +=1
            if b2<=a2:
                lowerTriListA[i] = a2
                lowerTriListB[i] = b2
                
                upperTriListX[i] = b2
                upperTriListY[i] = a2
            else:
                lowerTriListA[i] = b2
                lowerTriListB[i] = a2
                
                upperTriListX[i] = a2
                upperTriListY[i] = b2
        elif a3 <=1 and b3 <=1:
            # count+=1
            if b3<=a3:
                lowerTriListA[i] = a3
                lowerTriListB[i] = b3
                
                upperTriListX[i] = b3
                upperTriListY[i] = a3
            else:
                lowerTriListA[i] = b3
                lowerTriListB[i] = a3
                
                upperTriListX[i] = a3
                upperTriListY[i] = b3
                
        # counts[i] = count
        # else:
        #     print(i)
        #     raise ValueError
    return lowerTriListA, lowerTriListB, upperTriListX, upperTriListY


sns.set(style="darkgrid")
sns.set(rc={'axes.facecolor':'0.96'})
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
mpl.rcParams["text.latex.preamble"] = mpl.rcParams["text.latex.preamble"] + r'\usepackage{xfrac}'

# CB91_Blue = 'darkblue'#'#2CBDFE'
# oxfordblue = "#061A40"
# CB91_Green = '#47DBCD'
# CB91_Pink = '#F3A0F2'
# CB91_Purple = '#9D2EC5'
# CB91_Violet = '#661D98'
# CB91_Amber = '#F5B14C'
# red = "#FC4445"
# newred = "#E4265C"
# flame = "#DD6031"

# color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
#                CB91_Purple,
#                 # CB91_Violet,
#                 'dodgerblue',
#                 'slategrey', newred]
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

dfO = pd.read_csv(dataLoc+dfname, 
                 index_col=False)

dfO["J12"] = np.abs(dfO["J12"])
dfO["J23"] = np.abs(dfO["J23"])
dfO["J31"] = np.abs(dfO["J31"])

dfO["J12/J23"] = dfO.J12 / dfO.J23
dfO["J31/J23"] = dfO.J31 / dfO.J23
dfO["J31/J12"] = dfO.J31 / dfO.J12
dfO["J23/J12"] = dfO.J23 / dfO.J12
dfO["J23/J31"] = dfO.J23 / dfO.J31
dfO["J12/J31"] = dfO.J12 / dfO.J31


"""
get point on lower triangle
"""
x = dfO["J12/J23"].to_numpy()
y = dfO["J31/J23"].to_numpy()
t = dfO["J23/J12"].to_numpy()
d = dfO["J31/J12"].to_numpy() 
s = dfO["J23/J31"].to_numpy() 
p = dfO["J12/J31"].to_numpy() 

lowerTriListA, lowerTriListB, upperTriListX, upperTriListY = ListRatiosInLowerTriangle(x, y, t, d, s, p)

dfO["LowerT.X"] = lowerTriListA
dfO["LowerT.Y"] = lowerTriListB
dfO["UpperT.X"] = upperTriListX
dfO["UpperT.Y"] = upperTriListY


# dfO = dfO.drop(dfO[(dfO['beta'] == 2) & (dfO['omega0'] < 4)].index)
dfO = dfO.round({'A2': 3, "A3":3 })

#%%


"""
Not sure what this part does actually!
"""
from itertools import cycle
darkblue = 'darkblue'#'#2CBDFE'
oxfordblue = "#061A40"
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
red = "#FC4445"
newred = "#E4265C"
flame = "#DD6031"


colourlist = [darkblue, CB91_Green, CB91_Pink, CB91_Purple, CB91_Violet , CB91_Amber, newred, flame]
colours = cycle(colourlist)

iterator = cycle(colourlist)


omegaMin = 0
omegaMax = 20
A2Min = 0
A2Max = 30
A3Min = 0
A3Max = 30


sz =10
# fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(sz,sz/3),
#                            constrained_layout=True, sharey=True, sharex=True)

ms = 1.5
fig, ax = plt.subplots(figsize=(6,6))

for alpha in [2]:
    for beta in [9]:#[2,3, 4, 5, 7, 9]:
        

        realOmegaMin = alpha*omegaMin
        realOmegaMax = alpha*omegaMax
        
        dfP = dfO[(dfO.beta == beta)
                  &(dfO.alpha == alpha)
                  &(dfO.omega0 <= omegaMax)
                  &(dfO.omega0 >= omegaMin)
                  &(dfO.A2 >= A2Min)
                  &(dfO.A2 <= A2Max)
                  &(dfO.A3 >= A3Min)
                  &(dfO.A3 <= A3Max)
                  # &(dfO["LowerT.X"]>0.92)
                  # &(dfO["LowerT.Y"]>0.42)
                  # &(dfO["LowerT.Y"]<0.5)
                  ]
        
        if not dfP.empty:
            colour = next(iterator)
            print(colour)
            # n1s = dfP.n1.values.tolist()
            # n2s = dfP.n2.values.tolist()
            # n3s = dfP.n3.values.tolist()
            
            # x = dfP["J12/J23"].to_numpy()
            # y = dfP["J31/J23"].to_numpy()
            
            xLT = dfP["LowerT.X"]
            yLT = dfP["LowerT.Y"] 
            # xUT = dfP["UpperT.X"]
            # yUT = dfP["UpperT.Y"]
            
            # dfTri = pd.DataFrame(columns = ["LowerT.X", "LowerT.Y", "UpperT.X", "UpperT.Y"])
            # dfTri["LowerT.X"] = xLT
            
            # xLT = xLT["LowerT.X">0,9]
            
            print(colour)
            # ax.plot(np.abs(x), np.abs(y), '.', label=r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r"$", markersize=ms, color = colour)
            # ax.plot(np.abs(y), np.abs(x), '.',  markersize=ms, color = colour)
            # ax.plot(np.abs(1/x), np.abs(y/x), '.',  markersize=ms, color = colour)
            # ax.plot(np.abs(y/x), np.abs(1/x), '.',  markersize=ms, color = colour)
            # ax.plot(np.abs(1/y), np.abs(x/y), '.', markersize=ms, color = colour)
            # ax.plot(np.abs(x/y), np.abs(1/y), '.',  markersize=ms, color = colour)
            
            
            # t = dfP["J23/J12"].to_numpy()
            # d = dfP["J31/J12"].to_numpy() 
            # s = dfP["J23/J31"].to_numpy() 
            # p = dfP["J12/J31"].to_numpy() 
            ax.plot(np.abs(xLT), np.abs(yLT), '.', label=r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r"$", markersize=ms, color = colour)
            # ax.plot(np.abs(xUT), np.abs(yUT), '.', label=r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r"$", markersize=ms, color = colour)
            # ax.plot(np.abs(y), np.abs(x), '.',  markersize=ms, color = colour)
            # ax.plot(np.abs(t), np.abs(d), '.',  markersize=ms, color = colour)
            # ax.plot(np.abs(d), np.abs(t), '.',  markersize=ms, color = colour)
            # ax.plot(np.abs(s), np.abs(p), '.', markersize=ms, color = colour)
            # ax.plot(np.abs(p), np.abs(s), '.',  markersize=ms, color = colour)
            
            # ax.set_ylabel(r"$J_{31}/J_{23}$", rotation=0, labelpad=10)
            # ax.set_xlabel(r"$J_{12}/J_{23}$")
            ax.set_ylim([0,1])
            ax.set_xlim([0,1])
            
            # ax[0].plot(np.abs(n1s), np.abs(n2s), '.', label=r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r"$")
            # ax[0].set_ylabel("n2", rotation=0, labelpad=10)
            # ax[0].set_xlabel("n1")
            
            # ax[1].plot(np.abs(n2s), np.abs(n3s), '.', label=r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r"$")
            # ax[1].set_ylim([0,5])
            # ax[1].set_ylabel("n3", rotation=0, labelpad=15)
            # ax[1].set_xlabel("n2")
            
            # ax[2].plot(np.abs(n3s), np.abs(n1s), '.', label=r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r"$")
            # ax[2].set_xlim([0,5])
            # ax[2].set_ylabel("n1", rotation=0, labelpad=15)
            # ax[2].set_xlabel("n3")

fig.suptitle(r"$\omega \in ["+str(omegaMin)+r", "+str(omegaMax)+r"], \> A_2 \in ["+str(A2Min)+r", "+str(A2Max)+r"],\>  A_3 \in ["+str(A3Min)+r", "+str(A3Max)+r"]$")
plt.legend(loc="upper right")
# fig.savefig(latexLoc+'Fig-n1n2n3.png', format='png', bbox_inches='tight')
# fig.savefig(latexLoc+'Fig-n1n2n3.pdf', format='pdf', bbox_inches='tight')

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
Plot showing values in lower triangle, non accumulative
"""
# dfN = dfO[(dfO["alpha"]==1)&(dfO["beta"]==2)&(dfO["omega"])]


saveFig = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs2/"
# for A2 in np.linspace(0,30,31):
for A2 in np.linspace(0,30,301):
    A2 = np.round(A2, 2)
    alpha = 1; beta = 2; 
    # omegaMax= 20; omegaMin = 0;
    omega0=10;
    # A2 = 19
    # A2Min = 0; A2Max = 30; A3Min = 0; A3Max = 30
    
    dfP = dfO[(dfO.beta == beta)
                      &(dfO.alpha == alpha)
                      &(dfO.omega0 == omega0)
                        &(dfO.A2 == A2)
                      ]
    
    dfP = dfP.sort_values(by=['A3'])
    
    
    xLT = dfP["LowerT.X"]
    yLT = dfP["LowerT.Y"] 
    
    
    
    
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(xLT, yLT, s=3, c=dfP.A3.to_numpy(), cmap="jet", marker=".")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    cbar = plt.colorbar(sc)
    plt.suptitle(r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r", \omega_0="+str(omega0)+r", A_2="+str(A2)+r"$")
    cbar.ax.set_ylabel(r"$A_3$", rotation=0, labelpad=10)
    # plt.savefig(saveFig+"alpha=1,beta=2,omega0=10,More,Accumulate/"+"Frame"+str(A2)+".png", format='png', bbox_inches='tight')
    plt.show()


#%%%


"""
Plot showing values in lower triangle accumulative
"""


saveFig = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs2/"


alpha = 1; beta = 2; 
# omegaMax= 20; omegaMin = 0;
omega0=10;
# A2 = 19
# A2Min = 0; A2Max = 30; A3Min = 0; A3Max = 30
# for A2 in np.linspace(0,30,31):
    

for i, A2max in enumerate(np.linspace(0,30,301)):
    # print(i)
    A2max =  np.round(A2max, 2)
    fig, ax = plt.subplots(figsize=(6,5))
    
    
    
    for A2 in np.linspace(0, A2max, i+1):
        
        # print("    ", A2)
        A2 =  np.round(A2, 2)
        
        dfP = dfO[(dfO.beta == beta)
                          &(dfO.alpha == alpha)
                          &(dfO.omega0 == omega0)
                            &(dfO.A2 == A2)
                          ]
        
        dfP = dfP.sort_values(by=['A3'])
        
        
        xLT = dfP["LowerT.X"]
        yLT = dfP["LowerT.Y"] 
        
        
        
        sc = ax.scatter(xLT, yLT, s=3, c=dfP.A3.to_numpy(), cmap="jet", marker=".")
        
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    cbar = plt.colorbar(sc)
    plt.suptitle(r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r", \omega_0="+str(omega0)+r", A_2="+str(A2max)+r"$")
    cbar.ax.set_ylabel(r"$A_3$", rotation=0, labelpad=10)
    # plt.savefig(saveFig+"alpha=1,beta=2,omega0=10,More,AccumulateR/"+"Frame"+str(A2max)+".png", format='png', bbox_inches='tight')
    plt.show()

