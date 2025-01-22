import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from floquet_simulations.utilities import PhaseShiftBetweenPlusMinusPi

def PlotParams(fontsize=10, serif_font="charter", mathtext_font="cm"): 
    # sns.set(style="darkgrid")
    # sns.set(rc={'axes.facecolor':'0.96'})
    """
    Mathtext fonts: (see https://matplotlib.org/stable/users/explain/text/mathtext.html)
    dejavusans: DejaVu Sans
    dejavuserif: DejaVu Serif
    cm: Computer Modern (TeX)
    stix: STIX (designed to blend well with Times)
    stixsans: STIX sans
    """
    # settings are for thesis
    params = {

              'legend.fontsize': fontsize*0.9,
              'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'xtick.labelsize': fontsize*0.9,
              'ytick.labelsize': fontsize*0.9,
              'font.size': fontsize,

              'xtick.bottom':True,
              'xtick.top':True,
              'ytick.left': True,
              'ytick.right':True,
              "xtick.direction": "in",
              "ytick.direction":"out", # because you dont want legend out
              "xtick.labeltop":False,
              "xtick.labelbottom":True,
              "ytick.labelleft":True,
              "ytick.labelright":False,
              "xtick.major.pad":2,
              "xtick.minor.pad":2,
              "ytick.major.pad":2,
              "ytick.minor.pad":2,
              "xtick.major.size":3,
              "ytick.major.size":3,

              "xtick.major.top":False,   # draw x axis top major ticks
              "xtick.major.bottom":  True,    # draw x axis bottom major ticks
              "ytick.major.left":True,   # draw x axis top major ticks
               "ytick.major.right":  False,    # draw x axis bottom major ticks
                #xtick.minor.top:     True    # draw x axis top minor ticks
                #xtick.minor.bottom:  True    # draw x axis bottom minor ticks

  
              'xtick.minor.visible': False,
              'axes.grid':False,

               "text.usetex": True,
              'font.family' : "serif",#'STIXGeneral',#"sans-serif",#"Arial"#    # this is for non math fonts
            "font.serif": serif_font,
            "mathtext.fontset": mathtext_font,# "dejavusans",  # Should be 'dejavusans' (default),
                               # 'dejavuserif', 'cm' (Computer Modern), 'stix',
                               # 'stixsans' or 'custom'
            "axes.formatter.use_mathtext": True,

            #   "font.sans-serif": "Computer Modern Roman",#"stix",#"Arial",

              #  'grid.alpha': 1,
              # 'grid.color': "0.9"
              # border around plot

               'axes.edgecolor' : "0.35",#'white',#"0.15",
                "axes.spines.left":   True,
                "axes.spines.bottom": True,
                "axes.spines.top":    True,
                "axes.spines.right":  True,


            #    "axes.facecolor": '0.97', #"white"
                # "axes.spines.left":   False,
            # "axes.spines.bottom": False,
            # "axes.spines.top":    False,
            # "axes.spines.right":  False,
            #             "axes.linewidth":1.25,
              }
    



    mpl.rcParams.update(params)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,nicefrac,xfrac}')

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


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def PlotAbsRealImagHamiltonian(HF,  figsize=(3,3), colourbar_pad=0.4, colourbar_size_percentage=5, save_location = False):
    absMax = np.max([np.abs(np.min(np.real(HF))),
                    np.abs(np.max(np.real(HF))),
                    np.abs(np.min(np.imag(HF))),
                    np.abs(np.max(np.imag(HF)))])

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    norm = mpl.colors.Normalize(vmin=-absMax, vmax=absMax)
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)

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

    plt.show()

def PlotAbsHamiltonian(HF, figsize=(3,3), colourbar_pad=0.4, colourbar_size_percentage=5, save_location = False, axes_tick_pos=False, axes_tick_labels=False, 
                         data_cmap_lims = (-1,1), colourbar_cmap_lims=(-1,1),  colourbar_ticks = [0]):


    norm = mpl.colors.Normalize(vmin=data_cmap_lims[0], vmax=data_cmap_lims[1])
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
    # 

    cmap = LinearSegmentedColormap.from_list('custom hamiltonians', ['#006F63', "#FFFFFF", '#F78320'], N=256)

    cm_unit = 1/2.54
    fig, ax = plt.subplots(constrained_layout=True, 
                           figsize=(figsize[0]*cm_unit, figsize[1]*cm_unit))
    pcm = ax.matshow(np.imag(HF), interpolation='none', cmap=cmap,  norm=norm)
    ax.set_title( r'$\mathrm{Imag}\left\{ {H_{S}^{t_0}}_{n,m} \right\}$')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xlabel('m')

    ax.set_ylabel('n', rotation=0, labelpad=10)
    if bool(axes_tick_pos):
        ax.set_xticks(axes_tick_pos)
        ax.set_yticks(axes_tick_pos)
    if bool(axes_tick_labels):
        ax.set_xticklabels(axes_tick_labels)
        ax.set_yticklabels(axes_tick_labels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=f"{colourbar_size_percentage}%", pad=colourbar_pad)

    if bool(colourbar_cmap_lims):
        new_norm = mpl.colors.Normalize(vmin=colourbar_cmap_lims[0], vmax=colourbar_cmap_lims[1])
        new_cmap = truncate_colormap(cmap, (colourbar_cmap_lims[0]-data_cmap_lims[0])/(data_cmap_lims[1] - data_cmap_lims[0]), (colourbar_cmap_lims[1]-data_cmap_lims[0])/(data_cmap_lims[1] - data_cmap_lims[0]))
        fig.colorbar(mpl.cm.ScalarMappable(norm=new_norm, cmap=new_cmap), cax=cax, ticks = colourbar_ticks)
    else:
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks = colourbar_ticks)


    if bool(save_location):
      if save_location.as_posix().find("pdf"):
        fig.savefig(save_location, format="pdf", bbox_inches="tight")
      elif save_location.as_posix().find("png"):
        print("A")
        fig.savefig(save_location, format="png", bbox_inches="tight")

    plt.show()



def PlotRealHamiltonian(HF, figsize=(3,3), colourbar_pad=0.4, colourbar_size_percentage=5, save_location = False, 
                        axes_tick_pos=False, axes_tick_labels=False, data_cmap_lims = (-1,1), 
                        colourbar_cmap_lims=(-1,1),  colourbar_ticks = np.arange(-1,1.2,0.2),
                        normaliser_type="linear",
                        title_labelpad=10, 
                        dpi=506):
    """
    cmap_lims is tuple giving range between -1 and 1, ie proportion of full colour map that is used. 
    """ 
    if normaliser_type =="linear":
        norm = mpl.colors.Normalize(vmin=data_cmap_lims[0], vmax=data_cmap_lims[1])
    elif normaliser_type == "log":
        linthresh = 1e-1
        norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)

    # 
    # 
    cmap = LinearSegmentedColormap.from_list('custom hamiltonians', ['#006F63', "#FFFFFF", '#F78320'], N=256)
    
    cm_unit = 1/2.54
    
    fig, ax = plt.subplots(constrained_layout=True, 
                           figsize=(figsize[0]*cm_unit, figsize[1]*cm_unit))
    ax.matshow(np.real(HF), interpolation='none', cmap=cmap,  norm=norm)
    ax.set_title( r'$ \left[H_{S}^{t_0}\right]_{i,j} /J$', pad=title_labelpad)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xlabel('$i$')
    ax.set_ylabel('$j$', rotation=0, labelpad=10)
    if bool(axes_tick_pos):
        ax.set_xticks(axes_tick_pos)
        ax.set_yticks(axes_tick_pos)
    if bool(axes_tick_labels):
        ax.set_xticklabels(axes_tick_labels)
        ax.set_yticklabels(axes_tick_labels)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=f"{colourbar_size_percentage}%", pad=colourbar_pad)

    if bool(colourbar_cmap_lims):
        new_norm = mpl.colors.Normalize(vmin=colourbar_cmap_lims[0], vmax=colourbar_cmap_lims[1])
        new_cmap = truncate_colormap(cmap, (colourbar_cmap_lims[0]-data_cmap_lims[0])/(data_cmap_lims[1] - data_cmap_lims[0]), (colourbar_cmap_lims[1]-data_cmap_lims[0])/(data_cmap_lims[1] - data_cmap_lims[0]))
        fig.colorbar(mpl.cm.ScalarMappable(norm=new_norm, cmap=new_cmap), cax=cax, ticks = colourbar_ticks)
    else:
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks = colourbar_ticks)
       
    if bool(save_location):
      if save_location.as_posix().find("pdf") >=0:
        fig.savefig(save_location, format="pdf", bbox_inches="tight")
      elif save_location.as_posix().find("png")>=0:
        fig.savefig(save_location, format="png", bbox_inches="tight", dpi=dpi)

    plt.show()

# def PlotRealHamiltonianLog(HF, figsize=(3,3), colourbar_pad=0.4, colourbar_size_percentage=5, save_location = False, 
#                         axes_tick_pos=False, axes_tick_labels=False, data_cmap_lims = (-1,1), 
#                         colourbar_cmap_lims=(-1,1),  colourbar_ticks = np.arange(-1,1.2,0.2)):
#     """
#     cmap_lims is tuple giving range between -1 and 1, ie proportion of full colour map that is used. 
#     """ 

#     linthresh = 1e-1
#     norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)

#     # 
#     # 
#     cmap = LinearSegmentedColormap.from_list('custom hamiltonians', ['#006F63', "#FFFFFF", '#F78320'], N=256)
    
#     cm_unit = 1/2.54
    
#     fig, ax = plt.subplots(constrained_layout=True, 
#                            figsize=(figsize[0]*cm_unit, figsize[1]*cm_unit))
#     ax.matshow(np.real(HF), interpolation='none', cmap=cmap,  norm=norm)
#     ax.set_title( r'$\mathrm{Real}\left\{ {H_{S}^{t_0}}_{n,m} \right\}$')
#     ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
#       labeltop=False)  
#     ax.set_xlabel('m')
#     ax.set_ylabel('n', rotation=0, labelpad=10)
#     # if axes_tick_pos:
#     #     ax.set_xticks(axes_tick_pos)
#     #     ax.set_yticks(axes_tick_pos)
#     # if axes_tick_labels:
#     #     ax.set_xticklabels(axes_tick_labels)
#     #     ax.set_yticklabels(axes_tick_labels)
    
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size=f"{colourbar_size_percentage}%", pad=colourbar_pad)

#     # if colourbar_cmap_lims:
#     #     new_norm = mpl.colors.Normalize(vmin=colourbar_cmap_lims[0], vmax=colourbar_cmap_lims[1])
#     #     new_cmap = truncate_colormap(cmap, (colourbar_cmap_lims[0]-data_cmap_lims[0])/(data_cmap_lims[1] - data_cmap_lims[0]), (colourbar_cmap_lims[1]-data_cmap_lims[0])/(data_cmap_lims[1] - data_cmap_lims[0]))
#     #     fig.colorbar(mpl.cm.ScalarMappable(norm=new_norm, cmap=new_cmap), cax=cax, ticks = colourbar_ticks)
#     # else:
#     #     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks = colourbar_ticks)
       
#     if save_location:
#       if save_location.as_posix().find("pdf") >=0:
#         fig.savefig(save_location, format="pdf", bbox_inches="tight")
#       elif save_location.as_posix().find("png")>=0:
#         fig.savefig(save_location, format="png", bbox_inches="tight")

#     plt.show()

def PlotImagHamiltonian(HF, figsize=(3,3), colourbar_pad=0.4, colourbar_size_percentage=5, save_location = False, axes_tick_pos=False, axes_tick_labels=False, 
                         data_cmap_lims = (-1,1), colourbar_cmap_lims=(-1,1),  colourbar_ticks = [0]):


    norm = mpl.colors.Normalize(vmin=data_cmap_lims[0], vmax=data_cmap_lims[1])
    # linthresh = 1e-1
    # norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)
    # 

    cmap = LinearSegmentedColormap.from_list('custom hamiltonians', ['#006F63', "#FFFFFF", '#F78320'], N=256)

    cm_unit = 1/2.54
    fig, ax = plt.subplots(constrained_layout=True, 
                           figsize=(figsize[0]*cm_unit, figsize[1]*cm_unit))
    pcm = ax.matshow(np.imag(HF), interpolation='none', cmap=cmap,  norm=norm)
    ax.set_title( r'$\mathrm{Imag}\left\{ {H_{S}^{t_0}}_{n,m} \right\}$')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax.set_xlabel('m')

    ax.set_ylabel('n', rotation=0, labelpad=10)
    if bool(axes_tick_pos):
        ax.set_xticks(axes_tick_pos)
        ax.set_yticks(axes_tick_pos)
    if bool(axes_tick_labels):
        ax.set_xticklabels(axes_tick_labels)
        ax.set_yticklabels(axes_tick_labels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=f"{colourbar_size_percentage}%", pad=colourbar_pad)

    if bool(colourbar_cmap_lims):
        new_norm = mpl.colors.Normalize(vmin=colourbar_cmap_lims[0], vmax=colourbar_cmap_lims[1])
        new_cmap = truncate_colormap(cmap, (colourbar_cmap_lims[0]-data_cmap_lims[0])/(data_cmap_lims[1] - data_cmap_lims[0]), (colourbar_cmap_lims[1]-data_cmap_lims[0])/(data_cmap_lims[1] - data_cmap_lims[0]))
        fig.colorbar(mpl.cm.ScalarMappable(norm=new_norm, cmap=new_cmap), cax=cax, ticks = colourbar_ticks)
    else:
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks = colourbar_ticks)


    if bool(save_location):
      if save_location.as_posix().find("pdf") >=0:
        fig.savefig(save_location, format="pdf", bbox_inches="tight")
      elif save_location.as_posix().find("png")>=0:
        fig.savefig(save_location, format="png", bbox_inches="tight")

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
    ax.set_title( r'$\mathrm{Angle}\{H_{n,m}^{\mathrm{t_0}}\}$')
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