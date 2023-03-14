
from scipy.special import jv
import numpy as np
import scipy.integrate as integrate
from numpy import pi, exp, sin, cos
import pandas as pd

from functions import CreateHFGeneralLoopA
from functions import Cosine,  RemoveWannierGauge

import sys


def ListRatioLowerTriangle(a1, b1, a2, b2, a3, b3):
    
    if a1 <=1 and b1 <=1:
        if b1<=a1:  # b is smaller than a so b is y and a is x
            lowerTriListX = a1
            lowerTriListY = b1
        else:       # a is smaller than b so a is y and b is x
            lowerTriListX = b1
            lowerTriListY = a1

    elif a2 <= 1 and b2 <=1:
        if b2 <=a2:
            lowerTriListX = a2 
            lowerTriListY = b2
        else:
            lowerTriListX = b2
            lowerTriListY = a2
    
    elif a3 <=1 and b3 <=1:
        if b3 <=a3:
            lowerTriListX = a3
            lowerTriListY = b3
        else:
            lowerTriListX = b3
            lowerTriListY = a3

    return lowerTriListX, lowerTriListY

if __name__ == "__main__":
    
   

    dfN = pd.DataFrame(columns=["A2", "A3", "omega0", "alpha", "beta",  "phi2/pi", "phi3rel/pi",
                                "FT-J12", "FT-J23", "FT-J31", 
                                "HE-J12", "HE-J23", "HE-J31",
                                "HE-O1", "HE-O2", "HE-O3",
				"FT-J12-ABS",
				"FT-J23-ABS",
				"FT-J31-ABS",
				"FT-Plaq-PHA",
				"HE-J12-ABS",
				"HE-J23-ABS",
				"HE-J31-ABS",
				"HE-Plaq-PHA",
				"FT-LowerT.X",
				"FT-LowerT.Y",
				"HE-LowerT.X",
				"HE-LowerT.Y"
		  ])
    
    dfN = dfN.astype({'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint32,
                              "beta":np.uint32,
                              "phi3/pi":np.float64,
                              "phi3rel/pi": np.float64,
                              "FT-J12":np.complex128,
                              "FT-J23":np.complex128,
                              "FT-J31":np.complex128,
                              "HE-J12":np.complex128,
                              "HE-J23":np.complex128,
                              "HE-J31":np.complex128,
                              "HE-O1":np.complex128,
                              "HE-O2":np.complex128,
                              "HE-O3":np.complex128,
			      "FT-J12-ABS":np.float64,
			      "FT-J23-ABS":np.float64,
			      "FT-J31-ABS":np.float64,
			      "FT-Plaq-PHA":np.float64,
			      "HE-J12-ABS":np.float64,
			      "HE-J23-ABS":np.float64,
			      "HE-J31-ABS":np.float64,
			      "HE-Plaq-PHA":np.float64,
			      "FT-LowerT.X":np.float64,
			      "FT-LowerT.Y":np.float64,
			      "HE-LowerT.X":np.float64,
			      "HE-LowerT.Y":np.float64
                              })
    
    
    
    A2 = BASHA2
    A3 = BASHA3
    
    alpha=1
    beta = 2
    # omega0 = np.float64(sys.argv[1])
    # omega0 = BASHOMEGA
    #omega0s = np.linspace(4,20,16*100+1)
    omega0s = np.linspace(4,20,16*10+1)
    
    
    
    dirname = "/rds/user/gmon2/hpc-work/"
    dfname = "TriangleRatios,alpha="+str(alpha)+",beta="+str(beta)+",A2="+str(A2)+",A3="+str(A3)+".csv"
    
    
    #phi3s = np.linspace(0, 2, 41)
    phi2s = np.linspace(0, 2, 21)
    phi3Rels = np.linspace(0, 2, 21) 
    # phi3s = [round(i,2) for i in phi3s]
    
    
    centres = [1,2]
    funcs = [Cosine, Cosine]
    
    onsite2 = 0
    onsite3 = 0
    
     
    i = len(dfN)
    
    for omega0 in omega0s:
        omega0 = np.round(omega0, 2)
                
        for phi2_frac in phi2s:
            phi2_frac = np.round(phi2_frac, 3)
                    
            phi2 = pi*phi2_frac
            
            for phi3_rel_frac in phi3Rels:
                
                phi3_rel_frac = np.round(phi3_rel_frac, 3)
                phi3_rel = pi*phi3_rel_frac
                phi3 = phi2 + phi3_rel
                omega2 = alpha*omega0
                omega3 = beta*omega0
                    
                T = 2*pi/omega0
                
                    
                    
                # first term expansion term
                J23_real = -(1/T)*integrate.quad(lambda t: cos(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t + phi2)), -T/2, T/2)[0]
                J23_imag = -1j*(1/T)*integrate.quad(lambda t: sin(A3/omega3*sin(omega3*t + phi3) - A2/omega2*sin(omega2*t + phi2)), -T/2, T/2)[0]
                        # we are removing esimate of absolute error
                J23 = J23_real + J23_imag
                        
                J31_real = -(1/T)*integrate.quad(lambda t: cos(-A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
                J31_imag = -1j*(1/T)*integrate.quad(lambda t: sin(-A3/omega3*sin(omega3*t + phi3)), -T/2, T/2)[0]
                J31 = J31_real + J31_imag
                    
                # J12 = -jv(0,A2/omega2)
                  
                J12_real = -(1/T)*integrate.quad(lambda t: cos(A2/omega2*sin(omega2*t + phi2)), -T/2, T/2)[0]
                J12_imag = -1j*(1/T)*integrate.quad(lambda t: sin(A2/omega2*sin(omega2*t + phi2)), -T/2, T/2)[0]
                J12 = J12_real + J12_imag
                
                
                        # HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])
                HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])
        
                        
                        #full Hamiltonian evolution
                paramss = [[A2, omega2, phi2, 0], [A3, omega3, phi2 + phi3_rel, 0]]
                _, HF_Ham = CreateHFGeneralLoopA(3, centres, funcs, paramss, T, 1)
                
                if not np.isnan(HF_Ham).all():
                            
                                
                    J12_FT = HF_FT[1,0] # should be real?
                    J23_FT = HF_FT[2,1] # should be real ?
                    J31_FT = HF_FT[0,2]
            
                    J12_Ham = HF_Ham[1][0] # should be real?
                    J23_Ham = HF_Ham[2][1]
                    J31_Ham = HF_Ham[0][2]
                            
                            
                    O1 = HF_Ham[0][0]
                    O2 = HF_Ham[1][1]
                    O3 = HF_Ham[2][2]
                   
    		 
                    for site in range(3):
                        HF_Ham = RemoveWannierGauge(HF_Ham, site, 3)
                        HF_FT = RemoveWannierGauge(HF_FT, site, 3)
                            
    
                    J12_FT_abs = np.abs(J12_FT)
                    J23_FT_abs = np.abs(J23_FT)
                    J31_FT_abs = np.abs(J31_FT)
                    J12_Ham_abs = np.abs(J12_Ham)  
                    J23_Ham_abs = np.abs(J23_Ham)
                    J31_Ham_abs = np.abs(J31_Ham)
                    
                    #J12_FT_pha = np.angle(-J12_FT)
                    #J23_FT_pha = np.angle(-J23_FT)
                    J31_FT_pha = np.angle(-HF_FT[0,2])
    		
            		#J12_Ham_pha = np.angle(-J12_Ham)
                    #J23_Ham_pha = np.angle(-J23_Ham)
                    J31_Ham_pha = np.angle(-HF_Ham[0,2])
    		
                    R1223_FT = J12_FT_abs/J23_FT_abs
                    R3123_FT = J31_FT_abs/J23_FT_abs
                    R3112_FT = J31_FT_abs/J12_FT_abs
                    R2312_FT = J23_FT_abs/J12_FT_abs
                    R1231_FT = J12_FT_abs/J31_FT_abs
                    R2331_FT = J23_FT_abs/J31_FT_abs
                    
                    R1223_Ham = J12_Ham_abs/J23_Ham_abs
                    R3123_Ham = J31_Ham_abs/J23_Ham_abs
                    R2312_Ham = J23_Ham_abs/J12_Ham_abs
                    R3112_Ham = J31_Ham_abs/J12_Ham_abs
                    R1231_Ham = J12_Ham_abs/J31_Ham_abs
                    R2331_Ham = J23_Ham_abs/J31_Ham_abs
                    
                    lowerTriangle_X_FT, lowerTriangle_Y_FT = ListRatioLowerTriangle(R1223_FT, R3123_FT, R2312_FT, R3112_FT, R1231_FT, R2331_FT)
                    lowerTriangle_X_Ham, lowerTriangle_Y_Ham = ListRatioLowerTriangle(R1223_Ham, R3123_Ham, R2312_Ham, R3112_Ham, R1231_Ham, R2331_Ham)
    
    		
    		
                    dfN.loc[i] = [np.float64(A2), np.float64(A3), np.float64(omega0), np.uint32(alpha), np.uint32(beta)
                                          , np.float64(phi2_frac), np.float64(phi3_rel_frac),
                                          np.complex128(J12_FT), np.complex128(J23_FT), np.complex128(J31_FT), 
                                          np.complex128(J12_Ham), np.complex128(J23_Ham), np.complex128(J31_Ham),
                                          np.complex128(O1), np.complex128(O2), np.complex128(O3),
    				      np.float64(J12_FT_abs), np.float64(J23_FT_abs), np.float64(J31_FT_abs), np.float64(J31_FT_pha),	
    				      np.float64(J12_Ham_abs), np.float64(J23_Ham_abs), np.float64(J31_Ham_abs), np.float64(J31_Ham_pha),
    				      np.float64(lowerTriangle_X_FT), np.float64(lowerTriangle_Y_FT),	
    				      np.float64(lowerTriangle_X_Ham), np.float64(lowerTriangle_Y_Ham)	
    					]
                            
                            
                            
                else:
                    dfN.loc[i] = [A2, A3, omega0, alpha, beta, phi2_frac, phi3_rel_frac,
                                          np.nan, np.nan, np.nan, 
                                          np.nan, np.nan, np.nan,
                                          np.nan, np.nan, np.nan,
                                          np.nan, np.nan, np.nan, np.nan,
                                          np.nan, np.nan, np.nan, np.nan,
                                          np.nan, np.nan, np.nan, np.nan]
                i +=1

    
    dfN['A2'] = dfN['A2'].apply(np.real)
    dfN['A3'] = dfN['A3'].apply(np.real)
    dfN['omega0'] = dfN['omega0'].apply(np.real)
    dfN['alpha'] = dfN['alpha'].apply(np.real)
    dfN['beta'] = dfN['beta'].apply(np.real)
    dfN['phi2/pi'] = dfN['phi2/pi'].apply(np.real)
    dfN['phi3rel/pi'] = dfN['phi3rel/pi'].apply(np.real)
    
    dfN = dfN.astype({
                'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint32,
                               "beta":np.uint32,
                               "phi2/pi":np.float64,
                               "phi3rel/pi":np.float64,
                             "FT-J12":np.complex128,
                             "FT-J23":np.complex128,
                             "FT-J31":np.complex128,
                             "HE-J12":np.complex128,
                             "HE-J23":np.complex128,
                             "HE-J31":np.complex128,
                             "HE-O1":np.complex128,
                             "HE-O2":np.complex128,
                             "HE-O3":np.complex128,
			     "FT-J12-ABS":np.float64,
			     "FT-J23-ABS":np.float64,
			     "FT-J31-ABS":np.float64,
			     "FT-Plaq-PHA":np.float64,
			     "HE-J12-ABS":np.float64,
			     "HE-J23-ABS":np.float64,
			     "HE-J31-ABS":np.float64,
			     "HE-Plaq-PHA":np.float64,
			     "FT-LowerT.X":np.float64,
			     "FT-LowerT.Y":np.float64,
			     "HE-LowerT.X":np.float64,
			     "HE-LowerT.Y":np.float64
                             })
            
    
   
    dfN.to_csv(dirname + dfname,index=False )



