import numpy as np
from numpy import pi

def PhaseShiftPositive(theta):
    theta = np.where(theta <0 , 2*pi+theta, theta); theta = np.where(theta>2*pi , theta -2*pi, theta) 
    return theta

def PhaseShiftBetweenPlusMinusPi(theta):
    theta = np.where(theta <-pi , 2*pi+theta, theta); theta = np.where(theta>pi , theta -2*pi, theta) 
    return theta
    
    