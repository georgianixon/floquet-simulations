
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
                              "alpha":np.uint8,
                              "beta":np.uint8,
                              "phi3/pi":np.float64,
                              #"phi3rel/pi":np.float64,
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

bigData.to_csv(dataLoc +  "Summary.csv",
                  index=False, 
                  )

# save phases data only
dfPha = bigData.drop(columns=[ 'FT-J12-ABS',
   'FT-J23-ABS', 'FT-J31-ABS', 'HE-J12-ABS', 'HE-J23-ABS',
   'HE-J31-ABS',  'FT-LowerT.X', 'FT-LowerT.Y',
   'HE-LowerT.X', 'HE-LowerT.Y'] )

# st = time.time()    
dfPha.to_csv(dataLoc + "Phases.csv", index=False)
# et = time.time()
# print("   save took", np.round(et - st, 1), "s")


# save lower triangle data only
dfLowerT = bigData.drop(columns=['FT-J12-ABS',
   'FT-J23-ABS', 'FT-J31-ABS', 'FT-Plaq-PHA', 'HE-J12-ABS', 'HE-J23-ABS',
   'HE-J31-ABS', 'HE-Plaq-PHA' ] )
# st = time.time()    
dfLowerT.to_csv(dataLoc + "LowerTriangle.csv", index=False)
# et = time.time()
# print("   save took", np.round(et - st, 1), "s")


# save HT data only
dfHE = bigData.drop(columns=['FT-J12-ABS',
   'FT-J23-ABS', 'FT-J31-ABS', 'FT-Plaq-PHA', 'FT-LowerT.X', 'FT-LowerT.Y' ] )
# st = time.time()    
dfHE.to_csv(dataLoc + "HE.csv", index=False)
# et = time.time()
# print("   save took", np.round(et - st, 1), "s")

HEMins = dfHE.drop(columns = ['HE-J12-ABS',
       'HE-J23-ABS', 'HE-J31-ABS'])
# st = time.time()    
HEMins.to_csv(dataLoc + "HE-Min.csv", index=False)
# et = time.time()
# print("   save took", np.round(et - st, 1), "s")




# save FT data only
dfFT = bigData.drop(columns=['HE-J12-ABS',
   'HE-J23-ABS', 'HE-J31-ABS', 'HE-Plaq-PHA', 'HE-LowerT.X', 'HE-LowerT.Y' ] )
# st = time.time()    
dfFT.to_csv(dataLoc + "FT.csv", index=False)
# et = time.time()
# print("   save took", np.round(et - st, 1), "s")
FTMins = dfHE.drop(columns = ['FT-J12-ABS',
       'FT-J23-ABS', 'FT-J31-ABS'])
FTMins.to_csv(dataLoc + "FT-Min.csv", index=False)
# et = time.time()
# print("   save took", np.round(et - st, 1), "s")
