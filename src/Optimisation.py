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
place = "Georgia Nixon"
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
 
    g
    
A3_start = 5
A2_start = 5
omega0_start = 19
phi3_frac_start = 0
alpha = 1
beta = 2

x_start = [A2_start, A3_start, omega0_start, phi3_frac_start]

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


bnds = ((0,30),(0,30),(4,20),(-2*pi,2*pi))
phaseTarget = random.random()*2
XTarget = random.random()
YTarget = random.random()*XTarget
T =  [XTarget, YTarget, phaseTarget]
print(T)

startHE = time.time()

solHE = minimize(lambda x: CostHamiltonianEvolution(T,x), x_start,
               options = {"disp":True},
                # method='Nelder-Mead',
               bounds=bnds)
endHE=time.time()


startFT = time.time()

solHE = minimize(lambda x: CostHamiltonianEvolution(T,x), x_start,
               options = {"disp":True},
                # method='Nelder-Mead',
               bounds=bnds)
endFT=time.time()


if solHE.success:
    print("Success! Took ", "{:.2f}".format(end-start),"s")
    x = sol.x
    A2_result = x[0]
    A3_result = x[1]
    omega0_result = x[2]
    phi3_frac_result = x[3]
    
    print("Starting guess:\tA2=",A2_start,"\t A3=",A3_start,"\t omega0=",omega0_start,"\tphi=",phi3_frac_start)
    
    print("Solution:\tA2=",A2_result,"\tA3=",A3_result,"\t omega=",omega0_result,"\t phi=",phi3_frac_result,"pi")
    print("Cost=",sol.fun)
    print("Desired results: \tX=",XTarget,"\tY=",YTarget,"\tphase=",phaseTarget)
    
    XResult, YResult, phaseResult = HamiltonianEvolution(x)
    print("Optimisation results: \tX=",XResult,"\tY=",YResult,"\tphase=",phaseResult)
else:
    print("Failure, took ",end-start,"s")
    
    
    #%%
# def Gradient(x, T):
    
#     dA2 = 0.01
#     dA3 = 0.01
#     domega0 = 0.01
#     dphi3_frac = 0.01
    
#     cost= Cost(T, x)
    
#     costdA2p = Cost(T, x+[dA2, 0, 0, 0])
#     costdA2m = Cost(T, x+[-dA2, 0, 0, 0])
    
#     costdA3p = Cost(T, x+[0, dA3, 0, 0])
#     costdA3m = Cost(T, x+[0, -dA3, 0, 0])
    
#     costdomega0p = Cost(T, x+[0, 0, domega0, 0])
#     costdomega0m = Cost(T, x+[0, 0, -domega0, 0])
    
#     costdphi3p = Cost(T, x+[0, 0, 0, 0])
#     costdphi3m = Cost(T, x+[0, 0, 0, 0])
    
    
    
    
    


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





#%%


#%%

import numpy as np

def f(x):
    return -np.exp(-(x - 0.7)**2)
result1 = optimize.minimize_scalar(f)
restult2 = optimize.minimize(f, [2,-1], method="Newton-CG")  

result1.success
x_min = result1.x
x_min

def test_minimize_l_bfgs_b_ftol(self):
    # Check that the `ftol` parameter in l_bfgs_b works as expected
    v0 = None
    for tol in [1e-1, 1e-4, 1e-7, 1e-10]:
        opts = {'disp': False, 'maxiter': self.maxiter, 'ftol': tol}
        sol = optimize.minimize(self.func, self.startparams,
                                method='L-BFGS-B', jac=self.grad,
                                options=opts)
        v = self.func(sol.x)

        if v0 is None:
            v0 = v
        else:
            assert_(v < v0)

        assert_allclose(v, self.func(self.solution), rtol=tol) 

#%%





