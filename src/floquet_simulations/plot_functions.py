import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def PlotParams(fontsize=12):
    params = {
                'legend.fontsize': fontsize,
    #          'figure.figsize': (20,8),
              'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'xtick.labelsize': fontsize*0.9,
              'ytick.labelsize': fontsize*0.9,
              'font.size': fontsize,
              'xtick.bottom':True,
              'xtick.top':False,
              'ytick.left': True,
              'ytick.right':False,
              ## draw ticks on the left side
    #          'axes.titlepad': 25
              'axes.edgecolor' :'white',
              'xtick.minor.visible': False,
              'axes.grid':False,
              'font.family' : 'STIXGeneral',#"sans-serif",#"Arial"#
              "font.sans-serif":"stix",#"Arial",
               'mathtext.fontset':"stix"#"Arial"#'stix'
              }
    
    mpl.rcParams.update(params)


def PlotAbsRealImagHamiltonian(HF):
    absMax = np.max([np.abs(np.min(np.real(HF))),
                    np.abs(np.max(np.real(HF))),
                    np.abs(np.min(np.imag(HF))),
                    np.abs(np.max(np.imag(HF)))])

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    norm = mpl.colors.Normalize(vmin=-absMax, vmax=absMax)
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
    # 

    '''abs real imag'''

    apply = [
             np.abs, 
             np.real, np.imag]
    labels = [
              r'$\mathrm{Abs}\{G_{n,m}\}$', 
              r'$\mathrm{Re}\{G_{n,m}\}$',
              r'$\mathrm{Imag}\{G_{n,m}\}$'
              ]

    sz = 8
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                           figsize=(sz,sz/2))

    for n1, f in enumerate(apply):
        pcm = ax[n1].matshow(f(HF), interpolation='none', cmap='PuOr',  norm=norm)
        ax[n1].set_title(labels[n1])
        ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[n1].set_xlabel('m')

    ax[0].set_ylabel('n', rotation=0, labelpad=10)

    # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    cax = plt.axes([1.03, 0.2, 0.03, 0.6])
    # fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
    fig.colorbar(pcm, cax=cax)
    plt.show()
    

def PlotRealHamiltonian(HF, figsize=(3,3)):
    absMax = np.max([np.abs(np.min(np.real(HF))),
                    np.abs(np.max(np.real(HF))),
                    np.abs(np.min(np.imag(HF))),
                    np.abs(np.max(np.imag(HF)))])

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    norm = mpl.colors.Normalize(vmin=-absMax, vmax=absMax)
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
    # 

    '''abs real imag'''

    # cmap = plt.cm.rainbow
    
    fig, ax = plt.subplots(constrained_layout=True, 
                           figsize=figsize)
    pcm = ax.matshow(np.real(HF), interpolation='none', cmap='PuOr',  norm=norm)
    ax.set_title( r'$\mathrm{Real}\{H_{n,m}^{\mathrm{eff}}\}$')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xlabel('m')

    ax.set_ylabel('n', rotation=0, labelpad=10)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.4)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="PuOr"), cax=cax)# label="unweighted graph distance")

    # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    # cax = plt.axes([0.95, 0.16, 0.06, 0.74])
    # fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
    # fig.colorbar(pcm, cax=cax)
    plt.show()