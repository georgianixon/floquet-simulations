# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:38:33 2020

@author: Georgia
"""

import matplotlib
import seaborn as sns

def PLOT():
    
    sns.set(style="darkgrid")
    sns.set(rc={'axes.facecolor':'0.95'})
#    sns.set_style({'grid.color': '0.85'})
#    sns.set_style( {'axes.axisbelow': False,
#                    'axes.edgecolor': '.8',
#                    'axes.facecolor': '0.95',
#                    'axes.grid': True,
#                    'axes.labelcolor': '.1',
#                    'axes.spines.bottom': False,
#                    'axes.spines.left': False,
#                    'axes.spines.right': False,
#                    'axes.spines.top': False,
#                    'figure.facecolor': 'white',
#                    'font.family': 'sans-serif',
#                    'font.sans-serif': 'sans-serif',
#                    'grid.color': '0.85',
#                    'grid.linestyle': '-',
#                    'image.cmap': 'rocket',
#                    'lines.solid_capstyle': 'round',
#                    'patch.edgecolor': 'w',
#                    'patch.force_edgecolor': True,
#                    'text.color': '.15',
#                    'xtick.bottom': False,
#                    'xtick.color': '.15',
#                    'xtick.direction': 'out',
#                    'ytick.color': '.15',
#                    'ytick.direction': 'out',
#                    'ytick.left': False})
    
#    {'font.size': 12.0,
# 'axes.labelsize': 12.0,
# 'axes.titlesize': 12.0,
# 'xtick.labelsize': 12.0,
# 'ytick.labelsize': 12.0,
# 'legend.fontsize': 10.0,
# 'axes.linewidth': 1.25,
# 'grid.linewidth': 1.0,
# 'lines.linewidth': 1.5,
# 'lines.markersize': 6.0,
# 'patch.linewidth': 1.0,
# 'xtick.major.width': 1.25,
# 'ytick.major.width': 1.25,
# 'xtick.minor.width': 1.0,
# 'ytick.minor.width': 1.0,
# 'xtick.major.size': 6.0,
# 'ytick.major.size': 6.0,
# 'xtick.minor.size': 4.0,
# 'ytick.minor.size': 4.0}
    
     #plt.style.use('grayscale')
     #plt.figure(figsize=(10,7))
#    matplotlib.style.use('default')
     
     
     
    
    matplotlib.rcParams['mathtext.fontset'] = 'cm' #latex style, cm?
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    
    font = {'family' : 'STIXGeneral', # latex style
    #        'weight' : 'bold',
            'size'   : 12}
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', titlesize=12, #fontsize of axes title, ) 
                  labelsize=12)     # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=12)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=12)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=10)    # legend fontsize
    matplotlib.rc('figure', titlesize=12)