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










#%%

import numpy as np
from scipy import optimize
def f(x):
    return -np.exp(-(x - 0.7)**2)
result1 = optimize.minimize_scalar(f)
restult2 = optimize.minimize(f, [2,-1], method="Newton-CG")  

result1.success
x_min = result1.x
x_min

#%%



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

def HamiltonianEvolution(x):
    
    A2, A3, omega0, phi3_frac = x
    
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
    

def FirstTerm(x):
    
    A2, A3, omega0, phi3_frac = x
    
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
 
    
    
A3 = 20
A2 = 30
omega0 = 10
phi3_frac = 0.2
alpha = 1
beta = 2

x = [A2, A3, omega0, phi3_frac]



HE_X, HE_Y, HE_P = HamiltonianEvolution(x)
FT_X, FT_Y, FT_P = FirstTerm(x)




#%%


def gradient(x, dx):
    
    HamiltonianEvolution(x)

start = [A2, A3, omega0, phi3_frac]



def gradient_descent(gradient, start, learn_rate, n_iter):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        vector += diff
    return vector








