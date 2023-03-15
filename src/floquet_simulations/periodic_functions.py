
import numpy as np
from numpy import sin, cos, pi

"""Ramp params"""
def Ramp(params, t): # ramp
    a = params[0]
    omega = params[1]
    phi = params[2]
    onsite = params[3]
    
    nCycle = np.floor(t*omega/2/pi + phi/2/pi)
    y = a*omega*t/2/pi + a*phi/2/pi - nCycle*a + onsite
    return y 

def RampHalf(params, t): # ramp
    a = params[0]
    omega = params[1]
    phi = params[2]
    onsite = params[3]
    
    nHalfCycle = np.floor(t*omega/pi + phi/pi)
    y = (a*omega*t/pi + a*phi/pi - nHalfCycle*a)*((nHalfCycle + 1 ) % 2) + onsite
    return y 


"""Blip params"""
def Blip(params, t):
    a = params[0]
    omega = params[1]
    phi = params[2]
    onsite = params[3]
    
    nHalfCycle = np.floor(t*omega/pi + phi/pi)
    y = a*sin(omega*t + phi)*((nHalfCycle+1) % 2) + onsite
    return y


"""Usual Cos shake"""
def Cosine(params, t):
    a = params[0]
    omega = params[1]
    phi = params[2]
    onsite = params[3]
    y = a*cos(omega*t + phi)+ onsite
    return y 

def Zero(params, t):
    return 0

def OnsiteOnly(onsite, t):
    return onsite

def DoubleCosine(params, t):
    a1 = params[0][0]; a2 = params[0][1]
    omega1 = params[1][0]; omega2 = params[1][1]
    phi1 = params[2][0]; phi2 = params[2][1]
    onsite = params[3]
    y = a1*cos(omega1*t + phi1) + a2*cos(omega2*t + phi2) + onsite
    return y
