import numpy as np
from math import pi
from scipy.optimize import minimize_scalar
from scipy.special import  jv



def GetPhiOffset(time1, timeOffset, omega1, omega2):
    time2 = time1+timeOffset
    
    omegaT = np.gcd(round(100*omega1), round(100*omega2))/100
    totalT = 2*pi/omegaT
    
    phi1 = time1*omega1*totalT
    phi2 = time2*omega2*totalT
    
    return phi1, phi2, totalT


def RampGen(params, t): # ramp
    a = params[0]
    omega = params[1]
    phi = params[2]
    theta = params[4] 
    onsite = params[3]

    
    nCycles = np.floor(t*omega/2/pi)
    tCycle = t - nCycles*2*pi/omega
    
    multiplier_pre_phi = (np.sign(tCycle - phi/omega)%3)%2
    multiplier_post_theta =  (np.sign(-tCycle + phi/omega + theta/omega)%3)%2

    subtract_height = 2*a*(pi)/theta*nCycles
    y = (a*omega*t/theta - a*phi/theta - subtract_height)*multiplier_pre_phi*multiplier_post_theta + onsite
    return y


def ComputeAValsFromRequiredGradients(gradients):
    N = len(gradients)
    xvals = np.zeros(N)
    xzero = 2.4048
    for i, y in enumerate(gradients):
        if y > 0:
            sol = minimize_scalar(lambda x: np.abs(jv(0,x) - y),
                              bounds = (0,xzero),
                              method="bounded")
            xvals[i] = sol.x
        elif y < 0:
            sol = minimize_scalar(lambda x: np.abs(jv(0,x) - y),
                              bounds = (xzero, 3.8316),
                              method="bounded")
            xvals[i] = sol.x
    return xvals

#get A_vals
def GetAValsFromBesselXVals(bessel_x_vals, omega, addition_type = "accumulative", constant_shift=""):
    """make returning A Vals jump around 0 if accumulative = False
    Let A vals accumulate if accumulative = True
    Choose constant_shift to be one of 'zero centre', 'positive', or none 
    """
    
    if (addition_type != "accumulative") and (addition_type != "+2,-2") and (addition_type != "alternating"):
        raise TypeError("right type please")
        
    A_diff_vals = bessel_x_vals*omega
    A_vals = [0]
    for i, diff in enumerate(A_diff_vals):
        if addition_type == "accumulative":
            A_vals.append(A_vals[i] + diff)
        elif addition_type == "+2,-2":
            if (i %4 == 0) or (i%4 == 1):
                A_vals.append(A_vals[i] + diff)
            else:
                A_vals.append(A_vals[i] - diff)
        elif addition_type == "alternating":
            if i%2 == 0:
                A_vals.append(A_vals[i] + diff)
            else:
                A_vals.append(A_vals[i] - diff) 
    A_vals = np.array(A_vals)
    if constant_shift=="positive":
        A_vals_min = np.min(A_vals)
        A_vals = A_vals - A_vals_min
    elif constant_shift == "zero centre":
        A_vals_spread = np.max(A_vals) - np.min(A_vals)
        A_vals = A_vals - np.max(A_vals) + A_vals_spread/2
    return A_vals

