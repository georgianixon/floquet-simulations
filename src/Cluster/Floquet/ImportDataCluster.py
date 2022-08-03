
import numpy as np
import pandas as pd

dataLoc = "/rds/hpc-work/"

alpha = 1
beta = 2

dfs_full = []
dfs_no_raw = []

for A2 in np.linspace(0,30,31):
    
    for A3 in np.linspace(0,30,31):
        # print(str(int(A2)), str(int(A3)))
        
        fileName = "TriangleRatios,alpha="+str(int(alpha))+",beta="+str(int(beta))+",A2="+str(int(A2))+",A3="+str(int(A3))+".csv"
        print(fileName)
        try:
            dfi = pd.read_csv(dataLoc+fileName,
                          index_col=False)
            print(A2, A3)
            dfi_noraw = dfi.drop(columns=["FT-J12","FT-J23","FT-J31","HE-J12","HE-J23","HE-J31","HE-O1", "HE-O2","HE-O3"])
            dfi_noraw = dfi_noraw.astype({'A2': np.float64,
                              'A3': np.float64,
                              'omega0': np.float64,
                              "alpha":np.uint32,
                              "beta":np.uint32,
                              "phi2/pi":np.float64,
                              "phi3rel/pi":np.float64,
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
    
            dfs_no_raw.append(dfi_noraw)
            
        except:
            p = 0
            print("Oh no ", A2, A3)
            

bigData = pd.concat(dfs_no_raw, ignore_index=True)
#FInd rows with nans
nanRows = np.where(np.isnan(bigData["FT-J12-ABS"]))[0]
#drop nan locs
print("num of nans:", len(nanRows))
bigData = bigData.drop(nanRows)

bigData.to_csv(dataLoc +  "bigData.csv",
                  index=False, 
                  )