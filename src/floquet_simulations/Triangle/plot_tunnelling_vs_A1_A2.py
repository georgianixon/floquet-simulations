# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:29:18 2023
@author: GeorgiaNixon
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
place = "Georgia"
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# import seaborn as sns
import sys
from mpl_toolkits import mplot3d

# sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
# sys.path.append("/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Data/floquet-simulations-1/src/")
sys.path.append("//wsl$/Ubuntu-20.04/home/georgianixon/projects/floquet-simulations/src")


def Plot():
    # sns.set(style="darkgrid")
    # sns.set(rc={'axes.facecolor':'0.96'})
    size=18
    params = {
                'legend.fontsize': size*0.7,
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size*0.7,
              'ytick.labelsize': size*0.7,
              'font.size': size,
              'font.family': 'STIXGeneral',
    #          'axes.titlepad': 25,
              'mathtext.fontset': 'stix',
              
              # 'axes.facecolor': 'white',
              'axes.edgecolor': 'white',
              'axes.grid': True,
              'grid.alpha': 1,
              # 'grid.color': "0.9"
              "text.usetex": True
              }
    mpl.rcParams.update(params)
    mpl.rcParams["text.latex.preamble"] = mpl.rcParams["text.latex.preamble"] + r'\usepackage{xfrac}'
    CB91_Blue = 'darkblue'#'#2CBDFE'
    oxfordblue = "#061A40"
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    red = "#FC4445"
    newred = "#E4265C"
    flame = "#DD6031"
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                    CB91_Purple,
                    # CB91_Violet,
                    'dodgerblue',
                    'slategrey', newred]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
Plot()

def unique(a):
    unique, counts = np.unique(a, return_counts=True)
    return np.asarray((unique, counts)).T


def FloatToStringSave(a):
    return str(a).replace(".", "p")




dfO = pd.read_csv("D:Data/Set21-alpha=1,beta=2,omega=8,local/"+"data_3.csv",
                           index_col=False)
# dfO = pd.read_csv("D:/Data/Merges/alpha=1,beta=2,omega=8,0-40/FT/FT-Min,phi3=0.csv",
#                   index_col=False)
# dfP0 = pd.read_csv("D:/Data/Set12-alpha=1,beta=2,omega=8/Summaries/FT-ABS-phi3=0.csv", 
#                     index_col = False)
dfO = dfO.sort_values(by=['A3', 'A2'], ignore_index=True)



# domains
# dfP = dfO[(dfO.A3 <= 38.5)
#           &(dfO.A3 >=37)
#           &(dfO.A2 <=18.5)]
# n_section = 820
# A2 = np.resize(np.array(dfP.A2.to_list()), (n_section, n_section))
# A3 = np.resize(np.array(dfP.A3.to_list()), (n_section, n_section))
# X = np.resize(np.array(dfP["FT-LowerT.X"].to_list()), (n_section, n_section))
# Y = np.resize(np.array(dfP["FT-LowerT.Y"].to_list()), (n_section, n_section))


#%%
"""plot raw hopping"""
x = np.array(dfP0.A2.to_list())
y = np.array(dfP0.A3.to_list())
J12 = np.array(dfP0["FT-J12-ABS"].to_list())
J23 = np.array(dfP0["FT-J23-ABS"].to_list())
J31 = np.array(dfP0["FT-J31-ABS"].to_list())


# fig= plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(x, y, J12, cmap='viridis', edgecolor='none')
# plt.show()


fig= plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, J23, cmap='viridis', edgecolor='none')
plt.show()

# fig= plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(x, y, J31, cmap='viridis', edgecolor='none')
# plt.show()


#%%
"""plot just with J12, J23, J31"""

dfO["FT-J23oJ12"] = dfO["FT-J23-ABS"] / dfO["FT-J12-ABS"]
dfO["FT-J31oJ12"] = dfO["FT-J31-ABS"] / dfO["FT-J12-ABS"]

dfO["HE-J23oJ12"] = dfO["HE-J23-ABS"] / dfO["HE-J12-ABS"]
dfO["HE-J31oJ12"] = dfO["HE-J31-ABS"] / dfO["HE-J12-ABS"]
dfO = dfO.sort_values(by=['A3', 'A2'], ignore_index=True)


A2_min = 0
A2_max = 18.5
A2_range = int((A2_max - A2_min)*10 +1)
A3_min = 37
A3_max = 38.5
A3_range = int((A3_max - A3_min)*10 +1)
# domains
dfP = dfO[
           (dfO.A3 <=A3_max)
           &(dfO.A3 >= A3_min)
           &(dfO.A2 <=A2_max)
           &(dfO.A2 >=A2_min)
          ]
# dfP = dfO
n_section_x = A3_range#401#16; 
n_section_y = A2_range#301#186
A2_square = np.resize(np.array(dfP.A2.to_list()), (n_section_x, n_section_y))
A3_square = np.resize(np.array(dfP.A3.to_list()), (n_section_x, n_section_y))
X = np.resize(np.array(dfP["FT-J23oJ12"].to_list()), (n_section_x, n_section_y))
# X = np.resize(np.array(dfP["FT-LowerT.X"].to_list()), (n_section_x, n_section_y))
Y = np.resize(np.array(dfP["FT-J31oJ12"].to_list()), (n_section_x, n_section_y))
# Y = np.resize(np.array(dfP["FT-LowerT.Y"].to_list()), (n_section_x, n_section_y))




# fourth dimention - colormap
# create colormap according to x-value (can use any 50x50 array)
color_dimension = A3_square # change to desired fourth dimension
z_dimension = A2_square

minn, maxx = color_dimension.min(), color_dimension.max()
norm = mpl.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='cividis')
m.set_array([])
fcolors = m.to_rgba(color_dimension)


# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,z_dimension, 
                facecolors=fcolors,
                vmin=minn, vmax=maxx,
                rstride=1, cstride=1,  alpha=0.8, 
                shade=False,
                        linewidth=1,
                        antialiased=True
                        )
cset = ax.contourf(X, Y, z_dimension, zdir='z', offset=np.min(z_dimension),colors="0.7")
ax.zaxis.set_rotate_label(False) 
ax.yaxis.set_rotate_label(False) 
ax.xaxis.set_rotate_label(False) 
ax.set_xlabel(r'$J_{23}$', labelpad = 1, rotation=0)
ax.set_ylabel(r'$J_{31}$', labelpad = 4, rotation = 0)
ax.set_zlabel(r'$A_2$', labelpad = 8, rotation=0)
ax.tick_params(axis="x", pad = 0.001)
ax.tick_params(axis="y", pad = 0.001)
ax.set_xticks([0,1], labels=["0", r"$J_{12}$"])
ax.set_yticks([0,1], labels=["0", r"$J_{12}$"])
# ax.view_init(20,210 )
ax.view_init(20,250)
ax.set_xlim((0,1))
ax.set_ylim((0,1))
cbar = plt.colorbar(m)

cbar.ax.set_ylabel(r"$A_3$", rotation=0, labelpad=13)
# fig.canvas.show()
fig.show()
