import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from floquet_simulations.utilities import PhaseShiftBetweenPlusMinusPi

def PlotParams(fontsize=12, font="stix"):
    # sns.set(style="darkgrid")
    # sns.set(rc={'axes.facecolor':'0.96'})

    params = {
              'legend.fontsize': fontsize,
              'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'xtick.labelsize': fontsize*0.9,
              'ytick.labelsize': fontsize*0.9,
              'font.size': fontsize,
              'xtick.bottom':True,
              'xtick.top':False,
              'ytick.left': True,
              'ytick.right':False,

              'axes.edgecolor' :'white',#"0.15",
              'xtick.minor.visible': False,
              'axes.grid':False,
              'font.family' : 'STIXGeneral',#"sans-serif",#"Arial"#
              "font.sans-serif":"stix",#"Arial",
               'mathtext.fontset':"stix"#"Arial"#'stix'
              #  "text.usetex": True
              #  'grid.alpha': 1,
              # 'grid.color': "0.9"


            #    "axes.facecolor": '0.97', #"white"
            #     "axes.spines.left":   False,
            # "axes.spines.bottom": False,
            # "axes.spines.top":    False,
            # "axes.spines.right":  False,
            #             "axes.linewidth":1.25,
              }
    
    mpl.rcParams.update(params)

    # mpl.rcParams["text.latex.preamble"] = mpl.rcParams["text.latex.preamble"] + r'\usepackage{xfrac}'
    # CB91_Blue = 'darkblue'#'#2CBDFE'
    # oxfordblue = "#061A40"
    # CB91_Green = '#47DBCD'
    # CB91_Pink = '#F3A0F2'
    # CB91_Purple = '#9D2EC5'
    # CB91_Violet = '#661D98'
    # CB91_Amber = '#F5B14C'
    # red = "#FC4445"
    # newred = "#E4265C"
    # flame = "#DD6031"
    # color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
    #                 CB91_Purple,
    #                 # CB91_Violet,
    #                 'dodgerblue',
    #                 'slategrey', newred]
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


def PlotAbsRealImagHamiltonian(HF,  figsize=(3,3), colourbar_pad=0.4, colourbar_size_percentage=5, save_location = False):
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
    
    cmap = LinearSegmentedColormap.from_list('custom hamiltonians', ['#006F63', "#FFFFFF", '#F78320'], N=256)
    cm_unit = 1/2.54
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                            figsize=(figsize[0]*cm_unit, figsize[1]*cm_unit))

    for n1, f in enumerate(apply):
        pcm = ax[n1].matshow(f(HF), interpolation='none', cmap=cmap,  norm=norm)
        ax[n1].set_title(labels[n1])
        ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[n1].set_xlabel('m')

    ax[0].set_ylabel('n', rotation=0, labelpad=10)

    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size=f"{colourbar_size_percentage}%", pad=colourbar_pad)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)# label="unweighted graph distance")
    if save_location:
        if save_location.as_posix().find("pdf"):
            fig.savefig(save_location, format="pdf", bbox_inches="tight")
        elif save_location.as_posix().find("png"):
            fig.savefig(save_location, format="png", bbox_inches="tight")

    # # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    # cax = plt.axes([1.03, 0.2, 0.03, 0.6])
    # # fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
    # fig.colorbsar(pcm, cax=cax)
    plt.show()
    

def PlotRealHamiltonian(HF, figsize=(3,3), colourbar_pad=0.4, colourbar_size_percentage=5, save_location = False, axes_tick_pos=False, axes_tick_labels=False):
    absMax = np.max([np.abs(np.min(np.real(HF))),
                    np.abs(np.max(np.real(HF))),
                    np.abs(np.min(np.imag(HF))),
                    np.abs(np.max(np.imag(HF)))])

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    norm = mpl.colors.Normalize(vmin=-absMax, vmax=absMax)
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
    # 

    # cmap = plt.cm.rainbow
    # cmap= "PuOr"
    cmap = LinearSegmentedColormap.from_list('custom hamiltonians', ['#006F63', "#FFFFFF", '#F78320'], N=256)

    cm_unit = 1/2.54
    fig, ax = plt.subplots(constrained_layout=True, 
                           figsize=(figsize[0]*cm_unit, figsize[1]*cm_unit))
    pcm = ax.matshow(np.real(HF), interpolation='none', cmap=cmap,  norm=norm)
    ax.set_title( r'$\mathrm{Real}\{H_{n,m}^{\mathrm{eff}}\}$')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xlabel('m')
    ax.set_ylabel('n', rotation=0, labelpad=10)
    if axes_tick_pos:
        ax.set_xticks(axes_tick_pos)
        ax.set_yticks(axes_tick_pos)
    if axes_tick_labels:
        ax.set_xticklabels(axes_tick_labels)
        ax.set_yticklabels(axes_tick_labels)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=f"{colourbar_size_percentage}%", pad=colourbar_pad)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)# label="unweighted graph distance")
    if save_location:
      if save_location.as_posix().find("pdf"):
        fig.savefig(save_location, format="pdf", bbox_inches="tight")
      elif save_location.as_posix().find("png"):
        fig.savefig(save_location, format="png", bbox_inches="tight")
    # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    # cax = plt.axes([0.95, 0.16, 0.06, 0.74])
    # fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
    # fig.colorbar(pcm, cax=cax)
    plt.show()

def PlotImagHamiltonian(HF, figsize=(3,3), colourbar_pad=0.4, colourbar_size_percentage=5, save_location = False, axes_tick_pos=False, axes_tick_labels=False):
    absMax = np.max([np.abs(np.min(np.real(HF))),
                    np.abs(np.max(np.real(HF))),
                    np.abs(np.min(np.imag(HF))),
                    np.abs(np.max(np.imag(HF)))])

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    norm = mpl.colors.Normalize(vmin=-absMax, vmax=absMax)
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
    # 

    # cmap = plt.cm.rainbow
    # cmap= "PuOr"
    cmap = LinearSegmentedColormap.from_list('custom hamiltonians', ['#006F63', "#FFFFFF", '#F78320'], N=256)

    cm_unit = 1/2.54
    fig, ax = plt.subplots(constrained_layout=True, 
                           figsize=(figsize[0]*cm_unit, figsize[1]*cm_unit))
    pcm = ax.matshow(np.imag(HF), interpolation='none', cmap=cmap,  norm=norm)
    ax.set_title( r'$\mathrm{Imag}\{H_{n,m}^{\mathrm{eff}}\}$')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xlabel('m')

    ax.set_ylabel('n', rotation=0, labelpad=10)
    if axes_tick_pos:
        ax.set_xticks(axes_tick_pos)
        ax.set_yticks(axes_tick_pos)
    if axes_tick_labels:
        ax.set_xticklabels(axes_tick_labels)
        ax.set_yticklabels(axes_tick_labels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=f"{colourbar_size_percentage}%", pad=colourbar_pad)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)# label="unweighted graph distance")
    if save_location:
      if save_location.as_posix().find("pdf"):
        fig.savefig(save_location, format="pdf", bbox_inches="tight")
      elif save_location.as_posix().find("png"):
        fig.savefig(save_location, format="png", bbox_inches="tight")
    # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    # cax = plt.axes([0.95, 0.16, 0.06, 0.74])
    # fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
    # fig.colorbar(pcm, cax=cax)
    plt.show()

def PlotAngleHamiltonian(HF, figsize=(3,3), colourbar_pad=0.4, colourbar_size_percentage=5, save_location = False, axes_tick_pos=False, axes_tick_labels=False):

    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
    # 

    # cmap = plt.cm.rainbow
    # cmap= "PuOr"
    cmap = LinearSegmentedColormap.from_list('custom hamiltonians', ['#006F63', "#FFFFFF", '#F78320'], N=256)

    cm_unit = 1/2.54
    fig, ax = plt.subplots(constrained_layout=True, 
                           figsize=(figsize[0]*cm_unit, figsize[1]*cm_unit))
    pcm = ax.matshow(PhaseShiftBetweenPlusMinusPi(np.angle(HF)), interpolation='none', cmap=cmap,  norm=norm)
    ax.set_title( r'$\mathrm{Angle}\{H_{n,m}^{\mathrm{eff}}\}$')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xlabel('m')
    ax.set_ylabel('n', rotation=0, labelpad=10)
    if axes_tick_pos:
        ax.set_xticks(axes_tick_pos)
        ax.set_yticks(axes_tick_pos)
    if axes_tick_labels:
        ax.set_xticklabels(axes_tick_labels)
        ax.set_yticklabels(axes_tick_labels)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=f"{colourbar_size_percentage}%", pad=colourbar_pad)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)# label="unweighted graph distance")
    if save_location:
      if save_location.as_posix().find("pdf"):
        fig.savefig(save_location, format="pdf", bbox_inches="tight")
      elif save_location.as_posix().find("png"):
        fig.savefig(save_location, format="png", bbox_inches="tight")
    # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    # cax = plt.axes([0.95, 0.16, 0.06, 0.74])
    # fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)
    # fig.colorbar(pcm, cax=cax)
    plt.show()