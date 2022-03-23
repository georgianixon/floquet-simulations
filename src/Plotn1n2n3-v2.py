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
place = "Georgia Nixon"
import matplotlib as mpl
import seaborn as sns
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
from hamiltonians import CreateHFGeneral
from hamiltonians import Cosine, ConvertComplex

dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"
latexLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Local Modulation Paper/Analytics/"
dfname = "TriangleRatios-phasedata-v3.csv"


def ListRatiosInLowerTriangle(lst1a,lst1b, lst2a,lst2b, lst3a,lst3b):
    """
    Go through (x1,y1), (x2,y2) (x3,y3) combinations and find the one in the bottom right triangle
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


dfO = pd.read_csv(dataLoc+dfname, 
                 index_col=False, 
                 converters={"FT-J12": ConvertComplex,
                            "FT-J23": ConvertComplex,
                            "FT-J31": ConvertComplex,
                            "HE-J12": ConvertComplex,
                            "HE-J23": ConvertComplex,
                            "HE-J31": ConvertComplex,
                            "HE-01": ConvertComplex,
                            "HE-02": ConvertComplex,
                            "HE-03": ConvertComplex
                            })


# G = dfO[(dfO["A2"]==30) &(dfO["A3"]==30)&(dfO["omega0"]==20)&(dfO["phi3/pi"]==0.6)]
# print(np.angle(G["HE-J31"].to_numpy()[0]), np.angle(G["FT-J31"].to_numpy()[0]))
# print(G["HE-J31-PHA"].to_numpy()[0], G["FT-J31-PHA"].to_numpy()[0])

# dfO["FT-J12"] = np.conj(dfO["FT-J12"]*-1)
# dfO["FT-J23"] = np.conj(dfO["FT-J23"]*-1)
# dfO["FT-J31"] = np.conj(dfO["FT-J31"]*-1)

dfO["FT-J12-ABS"] = np.abs(dfO["FT-J12"])
dfO["FT-J23-ABS"] = np.abs(dfO["FT-J23"])
dfO["FT-J31-ABS"] = np.abs(dfO["FT-J31"])

#negative to account for the fact that matrix element is negative
dfO["FT-J12-PHA"] = np.angle(-dfO["FT-J12"])
dfO["FT-J23-PHA"] = np.angle(-dfO["FT-J23"])
dfO["FT-J31-PHA"] = np.angle(-dfO["FT-J31"])


dfO["HE-J12-ABS"] = np.abs(dfO["HE-J12"])
dfO["HE-J23-ABS"] = np.abs(dfO["HE-J23"])
dfO["HE-J31-ABS"] = np.abs(dfO["HE-J31"])

#negative to account for the fact that matrix element is negative
dfO["HE-J12-PHA"] = np.angle(-dfO["HE-J12"])
dfO["HE-J23-PHA"] = np.angle(-dfO["HE-J23"])
dfO["HE-J31-PHA"] = np.angle(-dfO["HE-J31"])


dfO["FT-J12/J23-ABS"] = dfO["FT-J12-ABS"] / dfO["FT-J23-ABS"]
dfO["FT-J31/J23-ABS"] = dfO["FT-J31-ABS"] / dfO["FT-J23-ABS"]
dfO["FT-J31/J12-ABS"] = dfO["FT-J31-ABS"] / dfO["FT-J12-ABS"]
dfO["FT-J23/J12-ABS"] = dfO["FT-J23-ABS"] / dfO["FT-J12-ABS"]
dfO["FT-J23/J31-ABS"] = dfO["FT-J23-ABS"] / dfO["FT-J31-ABS"]
dfO["FT-J12/J31-ABS"] = dfO["FT-J12-ABS"] / dfO["FT-J31-ABS"]

# """
# get point on lower triangle
# """
x = dfO["FT-J12/J23-ABS"].to_numpy()
y = dfO["FT-J31/J23-ABS"].to_numpy()
t = dfO["FT-J23/J12-ABS"].to_numpy()
d = dfO["FT-J31/J12-ABS"].to_numpy() 
s = dfO["FT-J23/J31-ABS"].to_numpy() 
p = dfO["FT-J12/J31-ABS"].to_numpy() 

lowerTriListA, lowerTriListB, upperTriListX, upperTriListY = ListRatiosInLowerTriangle(x, y, t, d, s, p)

dfO["FT-LowerT.X"] = lowerTriListA
dfO["FT-LowerT.Y"] = lowerTriListB
dfO["FT-UpperT.X"] = upperTriListX
dfO["FT-UpperT.Y"] = upperTriListY


# dfO = dfO.drop(dfO[(dfO['beta'] == 2) & (dfO['omega0'] < 4)].index)
# dfO = dfO.round({'A2': 3, "A3":3 })

#%%
"""
Phases - keep phi constant, vary A's'
"""

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/phiI=0.6,A2=30/"

  # C:\Users\Georgia Nixon\OneDrive - University of Cambridge\MBQD\Figs\ShakingTriangle\Phases\V2\FirstTerm
titles = [
    "Ham Evolution", "First Term"]
    # saveAs = [latexLoc + "EffectivePhases,HamEvolve,ByOmega0,alpha=1,beta=2.pdf" , latexLoc + "EffectivePhases,FirstTerm,ByOmega0,alpha=1,beta=2.pdf" ]
    # saveAs = [latexLoc + "EffectivePhases,HamEvolve,ByPhi,alpha=1,beta=2.pdf" , latexLoc + "EffectivePhases,FirstTerm,ByPhi,alpha=1,beta=2.pdf" ]
folder = [
    "HamEvolve" , 
           "FirstTerm",]
# x = dfO["phi3/pi"].to_numpy()*pi

saveTit="Both"
entry = ["HE-J31-PHA", "FT-J31-PHA"]
colours = ["#D30C7B", "#1BB18C"]

phiFrac = 0.6
# for  title1, saveTit in zip( titles, folder):
    
for omega0 in np.linspace(4,20,17):
# for omega0 in [20]:

    dfP = dfO[(dfO["phi3/pi"]==phiFrac)&
              (dfO["omega0"]==omega0)
               &(dfO["A2"]==30)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    # title = title1 + r", $\alpha=1$, $\beta=2$, $\phi="+str(phiFrac)+r"\pi$, $A_2 = 30$, $\omega_0="+str(omega0)+r"$"
    title = r"$\alpha=1$, $\beta=2$, $\phi="+str(phiFrac)+r"\pi$, $A_2 = 30$, $\omega_0="+str(omega0)+r"$"
    
    for lab, column, co in zip(titles, entry, colours):
        
        data = dfP[column].to_numpy()
        # if column == "FT-J31-PHA":
        #     print(1)
        #     data = np.conj(data)
    
        x = dfP["A3"].to_numpy()
        # x = dfO["omega0"].to_numpy()
        
        

    
        
        # sc = ax.scatter(x, data, s=3, c=dfP.A2.to_numpy(), cmap="jet", marker=".")
        ax.plot(x, data,'.', ms=1, color=co, label=lab)
        # ax.plot(x, data, 'x', ms=1, )
        
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    # ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    # ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    # ax.set_xlabel(r"$\omega_0$")
    ax.set_xlabel(r"$A_3$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    # cbar = plt.colorbar(sc)
    # cbar.ax.set_ylabel(r"$A_2$", rotation=0, labelpad=10)
    # plt.savefig(saveTit, format="pdf", bbox_inches="tight")
    ax.set_title(title)
    plt.legend(loc="upper right")
    plt.savefig(phasesFigLoc+saveTit+"/"+saveTit+"Phases,alpha=1,beta=2,A2=30,phiI=0.6,omega0="+str(omega0)+".png", format='png', bbox_inches='tight')
    plt.show()
        
      

#%%
"""
Phases - vary phi
"""

phasesFigLoc = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/Phases/V2/omega0=5,A2=30/"


titles = ["Ham Evolution", "First Term"]
    # saveAs = [latexLoc + "EffectivePhases,HamEvolve,ByOmega0,alpha=1,beta=2.pdf" , latexLoc + "EffectivePhases,FirstTerm,ByOmega0,alpha=1,beta=2.pdf" ]
    # saveAs = [latexLoc + "EffectivePhases,HamEvolve,ByPhi,alpha=1,beta=2.pdf" , latexLoc + "EffectivePhases,FirstTerm,ByPhi,alpha=1,beta=2.pdf" ]
folder = ["HamEvolve" , 
          "FirstTerm" ]
# x = dfO["phi3/pi"].to_numpy()*pi

entry = ["HE-J31-PHA", "FT-J31-PHA"]

saveTit = "Both"

colours = ["#D30C7B", "#1BB18C"]
omega0 = 5
# for column, title1, saveTit in zip(entry, titles, folder):
    

for A3 in np.linspace(0,30, 31):

    dfP = dfO[(dfO["omega0"]==omega0)&
              (dfO["A2"]==30)&
               (dfO["A3"]==A3)
              ]
    
    fig, ax = plt.subplots(figsize=(5,5))
    # title = title1 + r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2=30$, $A_3="+str(A3)+r"$"
    title = r"$\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2=30$, $A_3="+str(A3)+r"$"
    for lab, column, co in zip(titles, entry, colours):
        data = dfP[column].to_numpy()

        x = dfP["phi3/pi"].to_numpy()*pi
        # x = dfO["omega0"].to_numpy()

    # sc = ax.scatter(x, data, s=3, c=dfP.A2.to_numpy(), cmap="jet", marker=".")
        ax.plot(x, data,'.', ms=1, color=co, label=lab)#"#D30C7B")
        
    ax.set_title(title)
    ax.set_ylabel(r"$\xi$", rotation=0)
    ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
    ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
    ax.set_xlabel(r"$\phi$")
    ax.set_ylim([-pi-0.1, pi+0.1])
    # cbar = plt.colorbar(sc)
    # cbar.ax.set_ylabel(r"$\phi$", rotation=0, labelpad=10)
    # plt.savefig(saveTit, format="pdf", bbox_inches="tight")
    plt.legend()
    plt.savefig(phasesFigLoc+saveTit+"/"+saveTit+"Phases,alpha=1,beta=2,A2=30,omega0="+str(omega0)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
    plt.show()
        



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
for A2 in np.linspace(0,30,7):#np.linspace(0,30,301):
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
    
    
    xLT = dfP["FT-LowerT.X"]
    yLT = dfP["FT-LowerT.Y"] 
    
    
    
    
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
Plot showing values in lower triangle accumulative - First Term
"""

saveFig = "C:/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Figs2/"


alpha = 1; beta = 2; 
# omegaMax= 20; omegaMin = 0;
omega0=10;
# A2 = 19
# A2Min = 0; A2Max = 30; A3Min = 0; A3Max = 30
# for A2 in np.linspace(0,30,31):
    

for i, A2max in enumerate(np.linspace(0,30,7)):#np.linspace(0,30,301)):
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
        
        
        xLT = dfP["FT-LowerT.X"]
        yLT = dfP["FT-LowerT.Y"] 
        
        
        
        sc = ax.scatter(xLT, yLT, s=3, c=dfP.A3.to_numpy(), cmap="jet", marker=".")
        
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    cbar = plt.colorbar(sc)
    plt.suptitle(r"$\alpha="+str(alpha)+r", \beta="+str(beta)+r", \omega_0="+str(omega0)+r", A_2="+str(A2max)+r"$")
    cbar.ax.set_ylabel(r"$A_3$", rotation=0, labelpad=10)
    # plt.savefig(saveFig+"alpha=1,beta=2,omega0=10,More,AccumulateR/"+"Frame"+str(A2max)+".png", format='png', bbox_inches='tight')
    plt.show()

