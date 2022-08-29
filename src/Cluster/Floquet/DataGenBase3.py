# -*- coding: utf-8 -*-


from scipy.special import jv
import numpy as np
import scipy.integrate as integrate
from numpy import pi, exp, sin, cos
import pandas as pd

# import sys
# sys.path.append("/Users/Georgia/Code/MBQD/floquet-simulations/src/Cluster/Floquet")
# import time 


from functions import CreateHFGeneralv3
from functions import Cosine,  RemoveWannierGauge



def IntegralTerm(A2, alpha, A3, beta, phi3_frac, omega0, l ):
    
    T = 2*pi/omega0
     # first term expansion term
    term_real = (1/T)*integrate.quad(lambda t: 
                    cos(
                        A3/(beta*omega0)*sin(beta*omega0*t + phi3_frac*pi) 
                        +A2/(alpha*omega0)*sin(alpha*omega0*t)
                        +omega0*l*t
                        ), 
                        -T/2, T/2
                        )[0]
        
    term_imag = 1j*(1/T)*integrate.quad(lambda t: 
                    sin(
                        A3/(beta*omega0)*sin(beta*omega0*t + phi3_frac*pi) 
                        +A2/(alpha*omega0)*sin(alpha*omega0*t)
                        +omega0*l*t
                        ), 
                        -T/2, T/2
                        )[0]
        
    return term_real+term_imag

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
    # st = time.time()
    
   

    dfN = pd.DataFrame(columns=["A2", "A3", "omega0", "alpha", "beta",  "phi3/pi", 
                                # "t0",
                                
				"FT-J12-ABS",
				"FT-J23-ABS",
				"FT-J31-ABS",
				"FT-Plaq-PHA",
# 				"ST-J12-ABS",
# 				"ST-J23-ABS",
# 				"ST-J31-ABS",
# 				"ST-Plaq-PHA",
				"HE-J12-ABS",
				"HE-J23-ABS",
				"HE-J31-ABS",
				"HE-Plaq-PHA",
				"FT-LowerT.X",
				"FT-LowerT.Y",
# 				"ST-LowerT.X",
# 				"ST-LowerT.Y",
				"HE-LowerT.X",
				"HE-LowerT.Y",
                "FT-J12", "FT-J23", "FT-J31", 
                                # "ST-J12", "ST-J23", "ST-J31", 
                                "HE-J12", "HE-J23", "HE-J31",
                                # "ST-O1", "ST-O2", "ST-O3",
                                "HE-O1", "HE-O2", "HE-O3",
		  ])
    
    dfN = dfN.astype({'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint8,
                              "beta":np.uint8,
                              "phi3/pi":np.float64,
                              # "t0":np.float64,
                         
			      "FT-J12-ABS":np.float64,
			      "FT-J23-ABS":np.float64,
			      "FT-J31-ABS":np.float64,
			      "FT-Plaq-PHA":np.float64,
# 			      "ST-J12-ABS":np.float64,
# 			      "ST-J23-ABS":np.float64,
# 			      "ST-J31-ABS":np.float64,
# 			      "ST-Plaq-PHA":np.float64,
			      "HE-J12-ABS":np.float64,
			      "HE-J23-ABS":np.float64,
			      "HE-J31-ABS":np.float64,
			      "HE-Plaq-PHA":np.float64,
			      "FT-LowerT.X":np.float64,
			      "FT-LowerT.Y":np.float64,
# 			      "ST-LowerT.X":np.float64,
# 			      "ST-LowerT.Y":np.float64,
			      "HE-LowerT.X":np.float64,
			      "HE-LowerT.Y":np.float64,
                  
                       "FT-J12":np.complex128,
                              "FT-J23":np.complex128,
                              "FT-J31":np.complex128,
                              # "ST-J12":np.complex128,
                              # "ST-J23":np.complex128,
                              # "ST-J31":np.complex128,
                              "HE-J12":np.complex128,
                              "HE-J23":np.complex128,
                              "HE-J31":np.complex128,
                              # "ST-O1":np.complex128,
                              # "ST-O2":np.complex128,
                              # "ST-O3":np.complex128,
                              "HE-O1":np.complex128,
                              "HE-O2":np.complex128,
                              "HE-O3":np.complex128
                              })
    
    
    
    A2_base = BASH2
    A3_base = BASH3
    
    A2_additions = np.linspace(0,1,50, endpoint=False)
    # A2_additions = np.linspace(0,1,2, endpoint=False)
    A3_additions = np.linspace(0,1,50, endpoint=False)
    # A3_additions = np.linspace(0,1,2, endpoint=False)
    
    # As = [(10,10),(15,15),(20,20) ,(15,25), (25,25), (30,30)]
    # As = [(30,30)]
    alpha = 1
    beta = 2
    # omega0 = np.float64(sys.argv[1])
    # omega0 = BASHOMEGA
    # omega0_additions = np.linspace(0,1,20, endpoint=False)
    # omega0s = np.linspace(4,20,1600+1)
    omega0 = BASH3
    omega2 = alpha*omega0
    omega3 = beta*omega0

    T = 2*pi/omega0

                    
    phi3s = np.linspace(0, 2, 41)
    # phi3s = np.linspace(0, 2, 5)
    # phi3_additions = np.linspace(0,1,20,endpoint=False)
    # phi3s = np.linspace(0, 2, 3)
    # phi3s = [round(i,2) for i in phi3s]
   
    
    
    dirname = "/rds/user/gmon2/hpc-work/"+str(int(omega0))+"/"
    dfname = ("TriangleRatios,alpha="+str(alpha)+",beta="+str(beta)
              +",A2="+str(A2_base)+",A3="+str(A3_base)
              +".csv")
    
    # t0s = np.linspace(0,1,21)
    
    centres = [1,2]
    funcs = [Cosine, Cosine]
    
    onsite2 = 0
    onsite3 = 0
    
     
    i = len(dfN)
    for A2_add in A2_additions:
        for A3_add in A3_additions:
        # for omega0 in omega0s:
           
            
            
            for phi3_frac in phi3s:
                phi3 = phi3_frac*pi
                # for t0 in t0s:
                # phi2_addition = alpha*omega0*t0
                # phi3_addition = beta*omega0*t0
            # for A2_add in A2_additions:
                A2 = np.round(A2_base + A2_add, 2)
        
            #     for A3_add in A3_additions:S
                A3 = np.round(A3_base + A3_add, 2)
                    
                    
            #         for phi3_frac in phi3s:
                # st_0 = time.time()
            #             phi3_frac = np.round(phi3_frac, 2)
            #             phi3 = pi*phi3_frac
                        
                J23 =  -IntegralTerm(-A2, alpha, A3, beta, phi3_frac, omega0, 0 )
                J31 = -jv(0,A3/omega3)
                J12 = -jv(0,A2/omega2)
                        
                        # HF_FT = np.array([[0,np.conj(J12), J31], [J12, 0, np.conj(J23)], [np.conj(J31), J23, 0]])
                HF_FT = np.array([[0,J12, J31], 
                                  [J12, 0, np.conj(J23)],
                                  [J31, J23, 0]])
        
                        
                        #full Hamiltonian evolution
                paramss = [[A2, omega2, 0, 0], [A3, omega3, phi3, 0]]
                _, HF_Ham = CreateHFGeneralv3(3, centres, funcs, paramss, T, 1)
                
                
                if not np.isnan(HF_Ham).all():
        
                
                    # #second order terms...
                    # second_order_terms = np.zeros((3,3))
                    # for l in range(1, 7):
                    #     B0 = IntegralTerm(0, alpha, A3, beta, phi3_frac, omega0, -l )
                    #     B1 = IntegralTerm(0, alpha, A3, beta, phi3_frac, omega0, l )
                    #     C0 = IntegralTerm(0, alpha, -A3, beta, phi3_frac, omega0, -l )
                    #     C1 = IntegralTerm(0, alpha, -A3, beta, phi3_frac, omega0, l )
                    #     D0 = IntegralTerm(A2, alpha, -A3, beta, phi3_frac, omega0, -l )
                    #     D1 = IntegralTerm(A2, alpha, -A3, beta, phi3_frac, omega0, l )
                    #     E0 = IntegralTerm(-A2, alpha, A3, beta, phi3_frac, omega0, -l )
                    #     E1 = IntegralTerm(-A2, alpha, A3, beta, phi3_frac, omega0, l )
                    #     Jl = jv(l, A2/omega2)
                    #     Nl = (-1)**l
                
                    #     HlHmlCommutator = np.array([
                    #         [C0*B1 - C1*B0, C0*E1 - C1*E0, Jl*(Nl*D1 - D0)],
                    #         [B1*D0 - B0*D1, D0*E1 - E0*D1, Jl*(C1 - Nl*C0)],
                    #         [Jl*(Nl*E0 - E1), Jl*(B0 - Nl*B1), B0*C1 - C0*B1 +E0*D1-D0*E1]
                    #         ])/l
                    #     # print(np.sum(np.abs(GetEvalsAndEvecsGen(HlHmlCommutator)[0])))
                        
                    #     second_order_terms = second_order_terms + HlHmlCommutator
                        
                    # second_order_terms = second_order_terms/omega0
                
                    # HF_ST = HF_FT + second_order_terms
                
                    J12_FT = HF_FT[1,0] # should be real?
                    J23_FT = HF_FT[2,1] # should be real ?
                    J31_FT = HF_FT[0,2]
            
                    # J12_ST = HF_ST[1,0] # should be real?
                    # J23_ST = HF_ST[2,1] # should be real ?
                    # J31_ST = HF_ST[0,2]
            
                    J12_Ham = HF_Ham[1][0] # should be real?
                    J23_Ham = HF_Ham[2][1]
                    J31_Ham = HF_Ham[0][2]
                            
                            
                    O1_Ham = HF_Ham[0][0]
                    O2_Ham = HF_Ham[1][1]
                    O3_Ham = HF_Ham[2][2]
                    
                    # O1_ST = HF_ST[0][0]
                    # O2_ST = HF_ST[1][1]
                    # O3_ST = HF_ST[2][2]
                    
                    
                   
        		 
                    for site in range(3):
                        HF_Ham = RemoveWannierGauge(HF_Ham, site, 3)
                        HF_FT = RemoveWannierGauge(HF_FT, site, 3)
                        # HF_ST = RemoveWannierGauge(HF_ST, site, 3)
                            
        
                    J12_FT_abs = np.abs(J12_FT)
                    J23_FT_abs = np.abs(J23_FT)
                    J31_FT_abs = np.abs(J31_FT)
                    
                    # J12_ST_abs = np.abs(J12_ST)
                    # J23_ST_abs = np.abs(J23_ST)
                    # J31_ST_abs = np.abs(J31_ST)
                    
                    J12_Ham_abs = np.abs(J12_Ham)  
                    J23_Ham_abs = np.abs(J23_Ham)
                    J31_Ham_abs = np.abs(J31_Ham)
                    
                    J31_FT_pha = np.angle(-HF_FT[0,2])
                    # J31_ST_pha = np.angle(-HF_ST[0,2])
                    J31_Ham_pha = np.angle(-HF_Ham[0,2])
        		
                    R1223_FT = J12_FT_abs/J23_FT_abs
                    R3123_FT = J31_FT_abs/J23_FT_abs
                    R3112_FT = J31_FT_abs/J12_FT_abs
                    R2312_FT = J23_FT_abs/J12_FT_abs
                    R1231_FT = J12_FT_abs/J31_FT_abs
                    R2331_FT = J23_FT_abs/J31_FT_abs
                    
                    # R1223_ST = J12_ST_abs/J23_ST_abs
                    # R3123_ST = J31_ST_abs/J23_ST_abs
                    # R3112_ST = J31_ST_abs/J12_ST_abs
                    # R2312_ST = J23_ST_abs/J12_ST_abs
                    # R1231_ST = J12_ST_abs/J31_ST_abs
                    # R2331_ST = J23_ST_abs/J31_ST_abs
                    
                    R1223_Ham = J12_Ham_abs/J23_Ham_abs
                    R3123_Ham = J31_Ham_abs/J23_Ham_abs
                    R2312_Ham = J23_Ham_abs/J12_Ham_abs
                    R3112_Ham = J31_Ham_abs/J12_Ham_abs
                    R1231_Ham = J12_Ham_abs/J31_Ham_abs
                    R2331_Ham = J23_Ham_abs/J31_Ham_abs
                    
                    lowerTriangle_X_FT, lowerTriangle_Y_FT = ListRatioLowerTriangle(R1223_FT, 
                            R3123_FT, R2312_FT, R3112_FT, R1231_FT, R2331_FT)
                    # lowerTriangle_X_ST, lowerTriangle_Y_ST = ListRatioLowerTriangle(R1223_ST, 
                    #         R3123_ST, R2312_ST, R3112_ST, R1231_ST, R2331_ST)
                    lowerTriangle_X_Ham, lowerTriangle_Y_Ham = ListRatioLowerTriangle(R1223_Ham, 
                            R3123_Ham, R2312_Ham, R3112_Ham, R1231_Ham, R2331_Ham)
                    
        		
                    dfN.loc[i] = [np.float64(A2), np.float64(A3), np.float64(omega0),
                                  np.uint8(alpha), np.uint8(beta),
                                  np.float64(phi3_frac), 
                                  # np.float64(t0),
        
                                          
        				    np.float64(J12_FT_abs), np.float64(J23_FT_abs),
                        np.float64(J31_FT_abs), np.float64(J31_FT_pha),	
        				    # np.float64(J12_ST_abs), np.float64(J23_ST_abs), 
                #         np.float64(J31_ST_abs), np.float64(J31_ST_pha),	
        				    np.float64(J12_Ham_abs), np.float64(J23_Ham_abs), 
                        np.float64(J31_Ham_abs), np.float64(J31_Ham_pha),
                        
        				    np.float64(lowerTriangle_X_FT), np.float64(lowerTriangle_Y_FT),	
        				    # np.float64(lowerTriangle_X_ST), np.float64(lowerTriangle_Y_ST),	
        				    np.float64(lowerTriangle_X_Ham), np.float64(lowerTriangle_Y_Ham),
                            
                    np.complex128(J12_FT), np.complex128(J23_FT), np.complex128(J31_FT), 
                    # np.complex128(J12_ST), np.complex128(J23_ST), np.complex128(J31_ST), 
                    np.complex128(J12_Ham), np.complex128(J23_Ham), np.complex128(J31_Ham),
                    # np.complex128(O1_ST), np.complex128(O2_ST), np.complex128(O3_ST),
                    np.complex128(O1_Ham), np.complex128(O2_Ham), np.complex128(O3_Ham)
        					]
                 
                            
                else:
                    dfN.loc[i] = [A2, A3, omega0, alpha, beta, phi3_frac,
                                          np.nan, np.nan, np.nan, 
                                          # np.nan, np.nan, np.nan,
                                          # np.nan, np.nan, np.nan,
                                          np.nan, np.nan, np.nan,
                                          np.nan, np.nan, np.nan,
                                          np.nan, np.nan, np.nan, np.nan,
                                          np.nan, np.nan, np.nan, np.nan,
                                          # np.nan, np.nan, np.nan, np.nan,
                                          # np.nan, np.nan,
                                          np.nan, np.nan,
                                          np.nan, np.nan]
                i +=1
                
                # et_0 = time.time()
                # print("   A2=",A2,",A3=",A3,"omega0=",omega0,"phi3=",np.round(phi3_frac,2),",time=", np.round(et_0 - st_0, 6), "s")
                    

    
    dfN['A2'] = dfN['A2'].apply(np.real)
    dfN['A3'] = dfN['A3'].apply(np.real)
    dfN['omega0'] = dfN['omega0'].apply(np.real)
    dfN['alpha'] = dfN['alpha'].apply(np.real)
    dfN['beta'] = dfN['beta'].apply(np.real)
    dfN['phi3/pi'] = dfN['phi3/pi'].apply(np.real)
    
    
    dfN = dfN.astype({
                'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint8,
                               "beta":np.uint8,
                               "phi3/pi":np.float64,
                               # "t0":np.float64,
                         
			     "FT-J12-ABS":np.float64,
			     "FT-J23-ABS":np.float64,
			     "FT-J31-ABS":np.float64,
			     "FT-Plaq-PHA":np.float64,
# 			     "ST-J12-ABS":np.float64,
# 			     "ST-J23-ABS":np.float64,
# 			     "ST-J31-ABS":np.float64,
# 			     "ST-Plaq-PHA":np.float64,
			     "HE-J12-ABS":np.float64,
			     "HE-J23-ABS":np.float64,
			     "HE-J31-ABS":np.float64,
			     "HE-Plaq-PHA":np.float64,
			     "FT-LowerT.X":np.float64,
			     "FT-LowerT.Y":np.float64,
# 			     "ST-LowerT.X":np.float64,
# 			     "ST-LowerT.Y":np.float64,
			     "HE-LowerT.X":np.float64,
			     "HE-LowerT.Y":np.float64,
                 
                     "FT-J12":np.complex128,
                             "FT-J23":np.complex128,
                             "FT-J31":np.complex128,
                             # "ST-J12":np.complex128,
                             # "ST-J23":np.complex128,
                             # "ST-J31":np.complex128,
                             "HE-J12":np.complex128,
                             "HE-J23":np.complex128,
                             "HE-J31":np.complex128,
                             # "ST-O1":np.complex128,
                             # "ST-O2":np.complex128,
                             # "ST-O3":np.complex128,
                             "HE-O1":np.complex128,
                             "HE-O2":np.complex128,
                             "HE-O3":np.complex128
                             })

                            
            
    # et = time.time()
    # print("script took", np.round(et - st, 1), "s")
   
    dfN.to_csv(dirname + dfname, index=False )

# #%%
 
# figloc =  "C:/Users/Georgia/OneDrive - University of Cambridge/MBQD/Figs/ShakingTriangle/"

# labels = ["Strob", "FT", "ST"]
# import matplotlib.pyplot as plt
# for (A2, A3) in As:
#     for omega0 in omega0s:



#         dfP = dfN[(dfN.A2==A2)&
#                   (dfN.A3==A3)&
#                   (dfN.omega0==omega0)
#                   ]
        
#         fig, ax = plt.subplots(figsize=(5,5))
#         # title = title1 + r", $\alpha=1$, $\beta=2$, $\omega_0="+str(omega0)+r"$, $A_2=30$, $A_3="+str(A3)+r"$"
#         title = r"$\alpha=1,\beta=2,\omega_0="+str(omega0)+r", A_2="+str(A2)+",A_3="+str(A3)+r"$"
#          # lab, column, co in zip(titles, entry, colours):
#         for column, label in zip(["HE-Plaq-PHA", "FT-Plaq-PHA", "ST-Plaq-PHA"], labels):
#             data = dfP[column].to_numpy()
    
#             x = dfP["t0"].to_numpy()
    
#             # sc = ax.scatter(x, data, s=3, c=dfP["A3"].to_numpy(), cmap="jet", marker=".")
#             ax.plot(x, data,  label=label)#"#D30C7B")
            
#         ax.set_title(title)
#         ax.set_ylabel(r"$\xi$", rotation=0)
#         ax.set_yticks([-pi,-pi/2, 0,pi/2, pi])
#         ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", '0',r"$\frac{\pi}{2}$", r"$\pi$"])
#         # ax.set_xticks([0,pi/2, pi, 3*pi/2, 2*pi])
#         # ax.set_xticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi$"])
#         ax.set_xlabel(r"$t_0/T$")
#         ax.set_ylim([-pi-0.1, pi+0.1])
#         # cbar = plt.colorbar(sc)
#         # cbar.ax.set_ylabel(r"$A_3$", rotation=0, labelpad=30)
#         plt.legend(loc="upper right")
#         plt.savefig(figloc+"Phases,alpha=1,beta=2,A2="+str(A2)+",omega0="+str(omega0)+",A3="+str(A3)+".png", format='png', bbox_inches='tight')
    
#         # plt.savefig(phase
#         plt.show()
