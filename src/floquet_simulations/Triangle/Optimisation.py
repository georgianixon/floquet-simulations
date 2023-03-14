# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:54:31 2022

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
from hamiltonians import Cosine, ConvertComplex, RemoveWannierGauge
import scipy.integrate as integrate
from scipy.optimize import minimize
import time
import random


sns.set(style="darkgrid")
sns.set(rc={'axes.facecolor':'0.96'})
# size=12
# params = {
#             'legend.fontsize': size*0.7,
#           'axes.labelsize': size,
#           'axes.titlesize': size,
#           'xtick.labelsize': size*0.7,
#           'ytick.labelsize': size*0.7,
#           'font.size': size,
#           'font.family': 'STIXGeneral',
# #          'axes.titlepad': 25,
#           'mathtext.fontset': 'stix',
          
#           # 'axes.facecolor': 'white',
#           'axes.edgecolor': 'white',
#           'axes.grid': True,
#           'grid.alpha': 1,
#           # 'grid.color': "0.9"
#           "text.usetex": True
#           }


# mpl.rcParams.update(params)
mpl.rcParams["text.latex.preamble"] = mpl.rcParams["text.latex.preamble"] + r'\usepackage{xfrac}'




dataLoc = "C:/Users/" + place + "/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations/"

def ListRatiosInLowerTriangleSingle(J12, J23, J31):
    """
    Go through (x1,y1), (x2,y2) (x3,y3) combinations and find the one in the bottom right triangle
    """
    
    a1 = J12/J23
    b1 = J31/J23
    
    a2 = J12/J31
    b2 = J23/J31
    
    a3 = J23/J12
    b3 = J31/J12

    if a1 <=1 and b1 <=1:
        # count +=1
        if b1<a1:
            lowerTriListA = a1
            lowerTriListB = b1
            
            # upperTriListX = b1
            # upperTriListY = a1
        else:
            lowerTriListA = b1
            lowerTriListB= a1
            
            # upperTriListX = a1
            # upperTriListY = b1
    elif a2 <=1 and b2 <=1:
        # count +=1
        if b2<=a2:
            lowerTriListA = a2
            lowerTriListB = b2
            
            # upperTriListX = b2
            # upperTriListY = a2
        else:
            lowerTriListA = b2
            lowerTriListB = a2
            
            # upperTriListX = a2
            # upperTriListY = b2
    elif a3 <=1 and b3 <=1:
        # count+=1
        if b3<=a3:
            lowerTriListA = a3
            lowerTriListB = b3
            
            # upperTriListX = b3
            # upperTriListY = a3
        else:
            lowerTriListA = b3
            lowerTriListB = a3
            
            # upperTriListX = a3
            # upperTriListY = b3
                
    return lowerTriListA, lowerTriListB #, upperTriListX, upperTriListY

def HamiltonianEvolution(params):
    
    A2, A3, omega0, phi3_frac = params
    
    phi3 = pi*phi3_frac

    alpha = 1
    beta = 2
    
    omega2 = alpha*omega0
    omega3 = beta*omega0
    
    T = 2*pi/omega0
    
    centres = [1,2]
    funcs = [Cosine, Cosine]
    
    #full Hamiltonian evolution
    paramss = [[A2, omega2, 0, 0], [A3, omega3, phi3, 0]]
    _, HF = CreateHFGeneral(3, centres, funcs, paramss, T, 1)
    for site in range(3):
        HF = RemoveWannierGauge(HF, site, 3)
        
    # return HF[1][0], HF[2][1], HF[0][2] # J12, J23, J31
    
    phase = np.angle(-HF[0][2])
    
    HE_lowerTriListA, HE_lowerTriListB = ListRatiosInLowerTriangleSingle(np.abs(HF[1][0]), np.abs(HF[2][1]), np.abs(HF[0][2]))
    
    return HE_lowerTriListA, HE_lowerTriListB, phase
    

def FirstTerm(params):
    
    A2, A3, omega0, phi3_frac = params
    
    phi3 = pi*phi3_frac

    alpha = 1
    beta = 2
    
    omega2 = alpha*omega0
    omega3 = beta*omega0
    
    T = 2*pi/omega0
    # first term expansion term
    J23_real = -(1/T)*integrate.quad(lambda t: cos(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
    J23_imag = -1j*(1/T)*integrate.quad(lambda t: sin(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
    # we are removing esimate of absolute error
    J23 = J23_real + J23_imag
    
    J31_real = -(1/T)*integrate.quad(lambda t: cos(-A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
    J31_imag = -1j*(1/T)*integrate.quad(lambda t: sin(-A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
    J31 = J31_real + J31_imag
    
    J12 = -jv(0,A2/omega2)
        
    
    HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])
    
    for site in range(3):
        # HF = RemoveWannierGauge(HF, site, 3)
        HF_FT = RemoveWannierGauge(HF_FT, site, 3)
    
    phase = np.angle(-HF_FT[0][2])
    
    # return HF_FT[1,0], HF_FT[2,1], HF_FT[0,2]
    FT_lowerTriListA, FT_lowerTriListB = ListRatiosInLowerTriangleSingle(np.abs(HF_FT[1][0]), np.abs(HF_FT[2][1]), np.abs(HF_FT[0][2]))
    
    return FT_lowerTriListA, FT_lowerTriListB, phase
 
    
 

def CostHamiltonianEvolution(T, x):
    """
    T i target set of variables [phaseTarget, XTarget, YTarget]
    x = [A2, A3, omega0, phi3_frac]
    """
    a = 10
    b = 10
    XTarget, YTarget, phaseTarget = T
    XGuess, YGuess, phaseGuess = HamiltonianEvolution(x)
    costFunc = b*(phaseGuess - phaseTarget)**2 + a*(XGuess - XTarget)**2 + a*(YGuess - YTarget)**2
    return costFunc


def CostFirstTerm(T, x):
    """
    T i target set of variables [phaseTarget, XTarget, YTarget]
    x = [A2, A3, omega0, phi3_frac]
    """
    a = 10
    b = 10
    XTarget, YTarget, phaseTarget = T
    XGuess, YGuess, phaseGuess = FirstTerm(x)
    costFunc = b*(phaseGuess - phaseTarget)**2 + a*(XGuess - XTarget)**2 + a*(YGuess - YTarget)**2
    return costFunc

dfname = "OptimisationRandGuesses.csv"


dfC = pd.read_csv(dataLoc+dfname, 
                  index_col=False, 
                    # converters={"FT-J12": ConvertComplex,
                    #           "FT-J23": ConvertComplex,
                    #           "FT-J31": ConvertComplex,
                    #           "HE-J12": ConvertComplex,
                    #           "HE-J23": ConvertComplex,
                    #           "HE-J31": ConvertComplex,
                    #           "HE-O1": ConvertComplex,
                    #           "HE-O2": ConvertComplex,
                    #           "HE-O3": ConvertComplex
                    #             }
                  )


# dfC = pd.DataFrame(columns=["XTarget", "YTarget", "PhaseTarget", "A2Start", "A3Start", "Omega0Start", "Phi3FracStart", 
#                            "Tol", "OptimizerMethod", "Function", "Success", "Time", "Cost", 
#                             "A2Result", "A3Result", "Omega0Result", "Phi3FracResult", "XResult", "YResult", "PhaseResult"])

dfC = dfC.astype({"XTarget":np.float32,
                  "YTarget":np.float32,
                  "PhaseTarget": np.float32,
                  "A2Start": np.uint8,
                  "A3Start": np.uint8,
                  "Omega0Start": np.uint8,
                  "Phi3FracStart": np.uint8,
                  "Tol":np.float16,
                  "OptimizerMethod": str,
                  "Function": str,
                  "Success":np.bool_,
                  "Time":np.uint16,
                  "Cost":np.float32,
                  "A2Result": np.float32,
                  "A3Result": np.float32,
                  "Omega0Result":np.float32,
                  "Phi3FracResult": np.float32,
                  "XResult": np.float32,
                  "YResult":np.float32,
                  "PhaseResult":np.float32
                          })
    #%%
    
"""
Go through random X and Y target points on the lower triangle and a phase target point
See if we can find parameters to fit
"""

tolerence = 0.000001
i = len(dfC)
for i in range(1000):
    A3_start = 5
    A2_start = 5
    omega0_start = 19
    phi3_frac_start = 0
    alpha = 1
    beta = 2
    
    x_start = [A2_start, A3_start, omega0_start, phi3_frac_start]
    
    bnds = ((0,30),(0,30),(4,20),(-2*pi,2*pi))
    phase_target = np.angle(exp(1j*random.random()*2*pi))
    XTarget = random.random()
    YTarget = random.random()*XTarget
    T =  [XTarget, YTarget, phase_target]
    print("Target:\t",["{0:.4g}".format(i) for i in T])
    
    startHE = time.time()
    
    solHE = minimize(lambda x: CostHamiltonianEvolution(T,x), x_start,
                   options = {"disp":True},
                    method="L-BFGS-B",
                    tol=0.000001,
                   bounds=bnds)
    endHE=time.time()
    
    print("HE")
    if solHE.success:
        print("\tSuccess! Took ", "{:.2f}".format(endHE-startHE),"s")
        x = solHE.x
        A2_result = x[0]
        A3_result = x[1]
        omega0_result = x[2]
        phi3_frac_result = x[3]
        
        print("\tStarting guess:\tA2=",A2_start,"\t A3=",A3_start,"\t omega0=",omega0_start,"\tphi=",phi3_frac_start)
        
        print("\tSolution:\tA2=","{0:.4g}".format(A2_result),"\tA3=","{0:.4g}".format(A3_result),"\t omega=","{0:.4g}".format(omega0_result),"\t phi=","{0:.4g}".format(phi3_frac_result),"pi")
        print("\tCost=","{0:.4g}".format(solHE.fun))
        print("\tDesired results: \tX=","{0:.4g}".format(XTarget),"\tY=","{0:.4g}".format(YTarget),"\tphase=","{0:.4g}".format(phase_target))
        
        XResultHE, YResultHE, phaseResultHE = HamiltonianEvolution(x)
        # XResultFT, YResultFT, phaseResultFT = FirstTerm(x)
        print("\tOptimisation results HE: \tX=","{0:.4g}".format(XResultHE),"\tY=","{0:.4g}".format(YResultHE),"\tphase=","{0:.4g}".format(phaseResultHE))
        # print("\t[Optimisation results FT: \tX=","{0:.4g}".format(XResultFT),"\tY=","{0:.4g}".format(YResultFT),"\tphase=","{0:.4g}".format(phaseResultFT),"]")
        
        dfC.loc[i] = [XTarget, YTarget, phase_target, A2_start, A3_start, omega0_start, phi3_frac_start, 
                           tolerence, "Auto", "HE", 1, endHE-startHE, solHE.fun, 
                            A2_result, A3_result, omega0_result, phi3_frac_result, XResultHE, YResultHE, phaseResultHE]
    else:
        print("\tFailure, took ",endHE-startHE,"s")
        dfC.loc[i] = [XTarget, YTarget, phase_target, A2_start, A3_start, omega0_start, phi3_frac_start, 
                           tolerence, "L-BFGS-B", "HE", 0, endHE-startHE, solHE.fun, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    
dfC = dfC.astype({"XTarget":np.float32,
                  "YTarget":np.float32,
                  "PhaseTarget": np.float32,
                  "A2Start": np.uint8,
                  "A3Start": np.uint8,
                  "Omega0Start": np.uint8,
                  "Phi3FracStart": np.uint8,
                  "Tol":np.float16,
                  "OptimizerMethod": str,
                  "Function": str,
                  "Success":np.bool_,
                  "Time":np.uint16,
                  "Cost":np.float32,
                  "A2Result": np.float32,
                  "A3Result": np.float32,
                  "Omega0Result":np.float32,
                  "Phi3FracResult": np.float32,
                  "XResult": np.float32,
                  "YResult":np.float32,
                  "PhaseResult":np.float32
                         })


dfC.to_csv(dataLoc+dfname,
                  index=False, 
                  # columns=["A2", "A3", "omega0", "alpha", "beta", "J12", "J23", "J31"]
                  )


#%%

"""
Plot Results
"""

dfCC = pd.read_csv(dataLoc+dfname, 
                 index_col=False, 
                 )
dfCC = dfC.astype({"XTarget":np.float32,
                  "YTarget":np.float32,
                  "PhaseTarget": np.float32,
                  "A2Start": np.uint8,
                  "A3Start": np.uint8,
                  "Omega0Start": np.uint8,
                  "Phi3FracStart": np.uint8,
                  "Tol":np.float16,
                  "OptimizerMethod": str,
                  "Function": str,
                  "Success":np.bool_,
                  "Time":np.uint16,
                  "Cost":np.float32,
                  "A2Result": np.float32,
                  "A3Result": np.float32,
                  "Omega0Result":np.float32,
                  "Phi3FracResult": np.float32,
                  "XResult": np.float32,
                  "YResult":np.float32,
                  "PhaseResult":np.float32
                         })

#%%
"""
Plot Successes
"""

dfP = dfC[(dfC.Success ==1)&
          (dfC.Cost <=1)]
fig, ax = plt.subplots(figsize=(5,3))
sc = ax.scatter(dfP.XTarget, dfP.YTarget, c=dfP.Cost, s=1, cmap="jet", marker=".")
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlabel(r"$\frac{\mathrm{J}_a}{\mathrm{J}_c}$",  fontsize=14)
ax.set_ylabel(r"$\frac{\mathrm{J}_b}{\mathrm{J}_c}$", rotation = 0, labelpad=10, fontsize=14)
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r"Cost", rotation=0, labelpad=15)
ax.set_title(r"Cost by Hopping ratio position, where Cost <1")
plt.show()  


fig, ax = plt.subplots(figsize=(5,3))
sc = ax.scatter(dfP.PhaseTarget, dfP.Time, c=dfP.Cost, s=1, cmap="jet", marker=".")
ax.set_ylabel("Time (s)")
ax.set_xticks([-pi, -pi/2, 0,pi/2, pi])
ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r"Cost", rotation=0, labelpad=15)
ax.set_title(r"Cost by Phase position, where Cost <1")
plt.show()      



dfP = dfC[(dfC.Success ==1)&
          (dfC.Cost >=1)]
fig, ax = plt.subplots(figsize=(5,3))
sc = ax.scatter(dfP.XTarget, dfP.YTarget, c=dfP.Cost, s=1, cmap="jet", marker=".")
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlabel(r"$\frac{\mathrm{J}_a}{\mathrm{J}_c}$",  fontsize=14)
ax.set_ylabel(r"$\frac{\mathrm{J}_b}{\mathrm{J}_c}$", rotation = 0, labelpad=10, fontsize=14)
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r"Cost", rotation=0, labelpad=15)
ax.set_title(r"Cost by Hopping ratio position, where Cost >1")
plt.show()  


fig, ax = plt.subplots(figsize=(5,3))
sc = ax.scatter(dfP.PhaseTarget, dfP.Time, c=dfP.Cost, s=1, cmap="jet", marker=".")
ax.set_ylabel("Time (s)")
ax.set_xticks([-pi, -pi/2, 0,pi/2, pi])
ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r"Cost", rotation=0, labelpad=15)
ax.set_title(r"Cost by Phase position, where Cost >1")
plt.show()      

"""
Plot Failures
"""

dfP = dfC[(dfC.Success ==0)]
fig, ax = plt.subplots(figsize=(5,3))
sc = ax.scatter(dfP.XTarget, dfP.YTarget, c=dfP.Time, s=1, cmap="jet", marker=".")
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlabel(r"$\frac{\mathrm{J}_a}{\mathrm{J}_c}$",  fontsize=14)
ax.set_ylabel(r"$\frac{\mathrm{J}_b}{\mathrm{J}_c}$", rotation = 0, labelpad=10, fontsize=14)
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r"Time", rotation=0, labelpad=15)
ax.set_title(r"Failures by Hopping ratio position")
plt.show()  


fig, ax = plt.subplots(figsize=(5,3))
sc = ax.scatter(dfP.PhaseTarget, dfP.Time, c = "darkblue",  s=1, cmap="jet", marker=".")
# ax.set_yticks([0])
# ax.set_yticklabels([""])
ax.set_ylabel("Time (s)")
ax.set_xticks([-pi, -pi/2, 0,pi/2, pi])
ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_xlabel(r"effective $\phi$")
# cbar = plt.colorbar(sc)
# cbar.ax.set_ylabel(r"Time", rotation=0, labelpad=15)
ax.set_title(r"Failures by Phase position")
plt.show()     

 
#%%


startFT = time.time()

solFT = minimize(lambda x: CostFirstTerm(T,x), x_start,
               options = {"disp":True},
                # method='Nelder-Mead',
                # tol=0.01,
               bounds=bnds)
endFT=time.time()

print("FT")

if solFT.success:
    print("")
    print("\tSuccess! Took ", "{:.2f}".format(endFT-startFT),"s")
    x = solFT.x
    A2_result = x[0]
    A3_result = x[1]
    omega0_result = x[2]
    phi3_frac_result = x[3]
    
    print("\tStarting guess:\tA2=",A2_start,"\t A3=",A3_start,"\t omega0=",omega0_start,"\tphi=",phi3_frac_start)
    
    print("\tSolution:\tA2=","{0:.4g}".format(A2_result),"\tA3=","{0:.4g}".format(A3_result),"\t omega=","{0:.4g}".format(omega0_result),"\t phi=","{0:.4g}".format(phi3_frac_result),"pi")
    print("\tCost=","{0:.4g}".format(solFT.fun))
    print("\tDesired results: \tX=","{0:.4g}".format(XTarget),"\tY=","{0:.4g}".format(YTarget),"\tphase=","{0:.4g}".format(phase_target))
    
    XResultHE, YResultHE, phaseResultHE = HamiltonianEvolution(x)
    XResultFT, YResultFT, phaseResultFT = FirstTerm(x)
    print("\tOptimisation results FT: \tX=","{0:.4g}".format(XResultFT),"\tY=","{0:.4g}".format(YResultFT),"\tphase=","{0:.4g}".format(phaseResultFT))
    print("\t[Optimisation results HE: \tX=","{0:.4g}".format(XResultHE),"\tY=","{0:.4g}".format(YResultHE),"\tphase=","{0:.4g}".format(phaseResultHE),"]")
    
else:
    print("\tFailure, took ",endFT-startFT,"s")
    print("Try Nelder Mead")
    startFT = time.time()

    solFT = minimize(lambda x: CostFirstTerm(T,x), x_start,
                   options = {"disp":True},
                    method='Nelder-Mead',
                   # bounds=bnds
                   )
    endFT=time.time()
    
    if solFT.success:
        print("")
        print("\tSuccess Nelder Mead! Took ", "{:.2f}".format(endFT-startFT),"s")
        x = solFT.x
        A2_result = x[0]
        A3_result = x[1]
        omega0_result = x[2]
        phi3_frac_result = x[3]
        
        print("\tStarting guess:\tA2=",A2_start,"\t A3=",A3_start,"\t omega0=",omega0_start,"\tphi=",phi3_frac_start)
        
        print("\tSolution:\tA2=","{0:.4g}".format(A2_result),"\tA3=","{0:.4g}".format(A3_result),"\t omega=","{0:.4g}".format(omega0_result),"\t phi=","{0:.4g}".format(phi3_frac_result),"pi")
        print("\tCost=","{0:.4g}".format(solFT.fun))
        print("\tDesired results: \tX=","{0:.4g}".format(XTarget),"\tY=","{0:.4g}".format(YTarget),"\tphase=","{0:.4g}".format(phase_target))
        
        XResultHE, YResultHE, phaseResultHE = HamiltonianEvolution(x)
        XResultFT, YResultFT, phaseResultFT = FirstTerm(x)
        print("\tOptimisation results FT: \tX=","{0:.4g}".format(XResultFT),"\tY=","{0:.4g}".format(YResultFT),"\tphase=","{0:.4g}".format(phaseResultFT))
        print("\t[Optimisation results HE: \tX=","{0:.4g}".format(XResultHE),"\tY=","{0:.4g}".format(YResultHE),"\tphase=","{0:.4g}".format(phaseResultHE),"]")
        
    else:
        print("\tFailure again, took ",endFT-startFT,"s")






#%%



