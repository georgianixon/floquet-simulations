
from scipy.special import jv, jn_zeros
import numpy as np
import scipy.integrate as integrate
from numpy import pi, exp, sin, cos
import pandas as pd


import sys
path = "//wsl$/Ubuntu-20.04/home/georgianixon/projects/floquet-simulations/src"
sys.path.append(path)

save_path = "D:/Data/Set21-alpha=1,beta=2,omega=8,local/"
from hamiltonians import CreateHFGeneralLoopA
from hamiltonians import Cosine,  RemoveWannierGauge
   

dfN = pd.DataFrame(columns=["A2", "A3", "omega0", "alpha", "beta", "phi3rel/pi",
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
				"HE-Plaq-PHA"
		  ])

dfN = dfN.astype({'A2': np.float64,
                          'A3': np.float64,
                          'omega0': np.float64,
                          "alpha":np.uint32,
                          "beta":np.uint32,
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
			      "HE-Plaq-PHA":np.float64
                          })



for A2 in np.linspace(0, 30, 300+1):
    for A3 in np.linspace(0, 40, 400+1):
        
        A2 = float(f"{A2:.4f}")
        A3 = float(f"{A3:.4f}")
        print(A2, A3)
        
        alpha=1
        beta = 2
        omega0=8
        
        
        
        centres = [1,2]
        funcs = [Cosine, Cosine]
        
        onsite2 = 0
        onsite3 = 0
        
        
        i = len(dfN)

        phi3_rel = 0
        omega2 = alpha*omega0
        omega3 = beta*omega0
        T = 2*pi/omega0
        
                
            
        # first term expansion term
        J23_real = -(1/T)*integrate.quad(lambda t: cos(A3/omega3*sin(omega3*t + phi3_rel) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
        J23_imag = -1j*(1/T)*integrate.quad(lambda t: sin(A3/omega3*sin(omega3*t + phi3_rel) - A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
                # we are removing esimate of absolute error
        J23 = J23_real + J23_imag
                
        J31_real = -(1/T)*integrate.quad(lambda t: cos(-A3/omega3*sin(omega3*t + phi3_rel)), -T/2, T/2)[0]
        J31_imag = -1j*(1/T)*integrate.quad(lambda t: sin(-A3/omega3*sin(omega3*t + phi3_rel)), -T/2, T/2)[0]
        J31 = J31_real + J31_imag
            
        # J12 = -jv(0,A2/omega2)
            
        J12_real = -(1/T)*integrate.quad(lambda t: cos(A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
        J12_imag = -1j*(1/T)*integrate.quad(lambda t: sin(A2/omega2*sin(omega2*t)), -T/2, T/2)[0]
        J12 = J12_real + J12_imag
        
        
                # HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])
        HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])

                
                #full Hamiltonian evolution
        paramss = [[A2, omega2, 0, 0], [A3, omega3, phi3_rel, 0]]
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
            
            J31_FT_pha = np.angle(-HF_FT[0,2])
            J31_Ham_pha = np.angle(-HF_Ham[0,2])


            dfN.loc[i] = [np.float64(A2), np.float64(A3), np.float64(omega0), np.uint32(alpha), np.uint32(beta)
                                    ,  np.float64(phi3_rel),
                                    np.complex128(J12_FT), np.complex128(J23_FT), np.complex128(J31_FT), 
                                    np.complex128(J12_Ham), np.complex128(J23_Ham), np.complex128(J31_Ham),
                                    np.complex128(O1), np.complex128(O2), np.complex128(O3),
                    np.float64(J12_FT_abs), np.float64(J23_FT_abs), np.float64(J31_FT_abs), np.float64(J31_FT_pha),	
                    np.float64(J12_Ham_abs), np.float64(J23_Ham_abs), np.float64(J31_Ham_abs), np.float64(J31_Ham_pha)	
                ]
                    
                    
                    
        else:
            dfN.loc[i] = [A2, A3, omega0, alpha, beta,  phi3_rel,
                                    np.nan, np.nan, np.nan, 
                                    np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan]
        i +=1

        
    # dfN['A2'] = dfN['A2'].apply(np.real)
    # dfN['A3'] = dfN['A3'].apply(np.real)
    # dfN['omega0'] = dfN['omega0'].apply(np.real)
    # dfN['alpha'] = dfN['alpha'].apply(np.real)
    # dfN['beta'] = dfN['beta'].apply(np.real)
    # dfN['phi3rel/pi'] = dfN['phi3rel/pi'].apply(np.real)
    
    dfN = dfN.astype({
                'A2': np.float64,
                            'A3': np.float64,
                            'omega0': np.float64,
                            "alpha":np.uint32,
                            "beta":np.uint32,
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
                "HE-Plaq-PHA":np.float64
                            })
            
    
    
    dfN.to_csv(save_path + "data_3.csv",index=False )



