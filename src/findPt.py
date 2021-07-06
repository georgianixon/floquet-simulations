# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:53:18 2021

@author: Georgia
"""

place = "Georgia Nixon"
import sys
sys.path.append("/Users/" + place + "/Code/MBQD/floquet-simulations/src")
from hamiltonians import create_HF, hoppingHF, roundcomplex, GetEvalsAndEvecs, formatcomplex, OrderEvecs
from scipy.linalg import eig 
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import jn_zeros
from numpy import exp



size = 25
params = {
            # 'legend.fontsize': size*0.75,
#          'figure.figsize': (20,8),
           'axes.labelsize': size,
           'axes.titlesize': size,
           "axes.edgecolor": "0.15",
            "axes.linewidth":1.25,
           'xtick.labelsize': size,
           'ytick.labelsize': size,
          'font.size': size,
          'xtick.bottom':True,
          'xtick.top':False,
          'ytick.left': True,
          'ytick.right':False,
          ## draw ticks on the left side
#          'axes.titlepad': 25
          # 'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
           'axes.grid':False,
          'font.family' : 'STIXGeneral',
          'mathtext.fontset':'stix',
          "axes.facecolor": "0",#"1",#'0.97',
          "legend.facecolor":"0.9",
          "legend.fontsize":size,
           "axes.spines.left":   False,
            "axes.spines.bottom": False,
            "axes.spines.top":    False,
            "axes.spines.right":  False,
          }
mpl.rcParams.update(params)

def phistring(phi):
    if phi == 0:
        return "0"
    else:
        return  r'$\pi /$' + str(int(1/(phi/pi)))
    


#%%

"""
Comparing initial conditions
"""

"""
Get data
"""

N=49; centre=24; a=35;
# omega = 11
omega=a/jn_zeros(0,1)[0]
phis = [0, pi/2]

form='SS-p'
rtol=1e-11

_, HF1 = create_HF(form, rtol, N, centre, a, phis[0], omega)
_, HF2 = create_HF(form, rtol, N, centre, a, phis[1], omega)

_, hopping1 = hoppingHF(N, centre, a, omega, phis[0])
_, hopping2 = hoppingHF(N, centre, a, omega, phis[1])

evals1, evecs1 = GetEvalsAndEvecs(HF1)
evals2, evecs2 = GetEvalsAndEvecs(HF2)

#flip if needed
for i in range(N):
    #if one is negative of the other, could both start at zero
    if np.all(evecs1[:,i]==-evecs2[:,i]):
        evecs1[:,i] = -evecs1[:,i]
    # else, make them start at the same value
    elif evecs1[0,i] != evecs2[0,i]:
        frac = evecs1[0,i] / evecs2[0,i]
        evecs2[:,i] = evecs2[:,i] * frac
    # assert(np.all(np.imag(evecs1)==0))
    # assert(np.all(np.imag(evecs2)==0))
       


    
#%%

"""
Plot evecs
"""
plt.rcParams['axes.facecolor'] = "0"
func = np.real
colour1 = "dodgerblue"#"#613DC1"#
colour2 = "1"#"0.7"#'#9D2EC5'

sz = 4
num = 7
fig, ax = plt.subplots(nrows = num, ncols = num, sharex=True,
                       sharey=True,
                       figsize=(sz*num*1.6,sz*num))

for i in range(num):
    for j in range(num):
        evec1 = evecs1[:,num*i + j]
        evec2 = evecs2[:,num*i + j]

        ax[i,j].plot(range(N), func(evec1), color=colour1,
                     label=r"$\phi=$"+phistring(phis[0])+
                     ", hop="+formatcomplex(hopping1,3), 
                     linewidth = 5)
        ax[i,j].plot(range(N), func(evec2), color=colour2, 
                     label=r"$\phi=$"+phistring(phis[1])+
                     ", hop="+formatcomplex(hopping2,3))

handles_legend, labels_legend = ax[0,0].get_legend_handles_labels()    
fig.legend(handles_legend, labels_legend, loc="right")
fig.suptitle(func.__name__+"(evecs)\nN="
              +str(N)
              + ", a = "+str(a)+ r", $\omega = $" +"{:.2f}".format(omega)
              , y=0.92)
plt.show()


sz = 8
fig, ax = plt.subplots(figsize=(sz*1.4,sz))
ax.plot(range(N), np.real(evals1), 'x', markersize=10, color=colour1, 
        label="H1")
ax.plot(range(N), np.real(evals2),  'o', markersize = 3, color=colour2, 
        label="H2")
fig.legend()
# fig.suptitle(title, y=ypos)
plt.show()



#%%

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# linthresh = 1e-2
# norm=mpl.colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-1.0, vmax=1.0, base=10)


B = evecs2.flatten()
M = np.zeros((N*N, N*N), dtype=np.complex128)
for i in range(N):
    M[i*N:(i+1)*N, i*N:(i+1)*N] = evecs1.T

P = np.linalg.solve(M, B)
P = P.reshape(N,N)


'''abs real imag'''

apply = [np.abs, np.real, np.imag]
labels = [
          r'$\mathrm{Abs}\{P_{n,m}\}$', 
          r'$\mathrm{Re}\{P_{n,m}\}$',
          r'$\mathrm{Imag}\{P_{n,m}\}$'
          ]

sz = 20
fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))

for n1, f in enumerate(apply):
    pos = ax[n1].matshow(f(P), interpolation='none', cmap='PuOr', norm=norm)
    ax[n1].set_title(labels[n1], fontsize=25)
    ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
  
cax = plt.axes([1.03, 0.1, 0.03, 0.8])
fig.colorbar(pos, ax=ax[n1], cax = cax)
fig.suptitle(r"$P(\phi=$"+phistring(phis[0])
             +r"$ \rightarrow \phi = $"
             +phistring(phis[1])+r"$)$"
             +"\n"
             +"SS"+", N="+str(N) + r", $V(t) = $"
             + str(a)+r'$\cos( $'
             + "{:.2f}".format(omega)
             + r"$ t + \phi)$"
             + ', rtol='+str(rtol)
    , fontsize = 25, y=0.95)
plt.show()


#%%

"""
plot matrix elements of P(t) over \phi
"""


N=49; centre=24; a=35;
omega = 9.6
# omega=a/jn_zeros(0,1)[0]
phis = np.linspace(0, 2*pi, num=30, endpoint=True)

form='SS-p'
rtol=1e-11

_, HF0 = create_HF(form, rtol, N, centre, a,0, omega)
evals0, evecs0 = GetEvalsAndEvecs(HF0)


#order evecs to make first nonzero element real and positive 
evecs0 = OrderEvecs(evecs0, N)
evecs0_R = roundcomplex(evecs0, 5)

# create M with information about initial state
# then M.X = B; B has info about final state; X gives info for P(t)
M = np.zeros((N*N, N*N), dtype=np.complex128)
for i in range(N):
    M[i*N:(i+1)*N, i*N:(i+1)*N] = evecs0.T


Ps = np.zeros((len(phis), N, N), dtype=np.complex128)
for i, phi in enumerate(phis):
    print("{:.2f}".format(phi))
    _, HFP = create_HF(form, rtol, N, centre, a,phi, omega)
    evalsP, evecsP = GetEvalsAndEvecs(HFP)
    evecsP_R = roundcomplex(evecsP, 5)

    # align evecsP
    # evecsP = AlignEvecs(evecs0, evecsP, N)
    for vec in range(N):

        #if one is negative of the other, for rounded evecs, flip one,
        #could both start at zero
        if np.all(evecs0_R[:,vec]==-evecsP_R[:,vec]):
            evecsP[:,vec] = -evecsP[:,vec]
            #redefine rounded evecsP
            evecsP_R[:,vec] = roundcomplex(evecsP[:,vec], 5)
        # else, make them start at the same value
        elif evecs0_R[0,vec] != evecsP_R[0,vec]:
            frac = evecs0[0,vec] / evecsP[0,vec]
            evecsP[:,vec] = evecsP[:,vec] * frac
            #redefine evecsP_R
            evecsP_R[:,vec] = roundcomplex(evecsP[:,vec], 5)
    
    # find P(t);  M.X = B; Find X
    B = evecsP.flatten()
    P = np.linalg.solve(M, B)
    P = P.reshape(N,N)
    Ps[i,:,:]=P
    


#%%

me1 = 22
me2 = 22
centreVals = Ps[:,me1,me2]


plt.rcParams['axes.facecolor'] = "0.95"
apply = [
         np.abs, 
         np.real, np.imag]

labels = [
          r"$\mathrm{Abs}\{P(t)_{" +str(me1)+r","+str(me2)+r"}\}$", 
          r"$\mathrm{Re}\{P(t)_{" +str(me1)+r","+str(me2)+r"}\}$",
          r"$\mathrm{Imag}\{P(t)_{" +str(me1)+r","+str(me2)+r"}\}$"
          ]

sz = 5
fig, ax = plt.subplots(ncols=3, nrows=1, sharey=True,  
                       constrained_layout=True, 
                       figsize=(sz*len(apply),sz))
for n1, f in enumerate(apply):
    pcm = ax[n1].plot(phis, f(centreVals), color="#D2419A")
    ax[n1].set_title(labels[n1])
    ax[n1].set_xlabel(r"$\phi$")

fig.suptitle(r"$P(t)_{"+str(me1)+r","+str(me2)+r"}$"+"\n"
    + r" $V(t) = |"+str(centre)+r"><"+str(centre)+r"|"
    + str(a)+r"\cos("
    + "{:.2f}".format(omega)
    + r"t + \phi"
    + r")$"
    + ', rtol='+str(rtol)
    +", N="+str(N)
    , fontsize = 25, y=1.2)

plt.show()    


"""
Phase Plot
"""
# me1 = 24
# me2 = 24
# centreVals = Ps[:,me1,me2]

sz = 6
fig, ax = plt.subplots(figsize=(sz*1.6,sz))

pcm = ax.plot(phis, np.unwrap(np.angle(centreVals)% 2*np.pi), color="#D2419A")
ax.set_xlabel(r"$\phi$")
ax.set_title(r"$\mathrm{Phase}\{P(t)_{" +str(me1)+r","+str(me2)+r"}\}$")
ax.set_ylim([-1,1])

fig.suptitle(r"$P(t)_{"+str(me1)+r","+str(me2)+r"}$"+"\n"
    + r" $V(t) = |"+str(centre)+r"><"+str(centre)+r"|"
    + str(a)+r"\cos("
    + "{:.2f}".format(omega)
    + r"t + \phi"
    + r")$"
    + ', rtol='+str(rtol)
    +", N="+str(N)
    , y=1.1)
    
plt.show()

sz = 6
fig, ax = plt.subplots(figsize=(sz*1.6,sz))

pcm = ax.plot(phis, np.unwrap(np.angle(centreVals)*2)/2, color="#D2419A")
ax.set_xlabel(r"$\phi$")
ax.set_title(r"$\mathrm{Phase}\{P(t)_{" +str(me1)+r","+str(me2)+r"}\}$")
ax.set_ylim([-1,1])

fig.suptitle(r"$P(t)_{"+str(me1)+r","+str(me2)+r"}$"+"\n"
    + r" $V(t) = |"+str(centre)+r"><"+str(centre)+r"|"
    + str(a)+r"\cos("
    + "{:.2f}".format(omega)
    + r"t + \phi"
    + r")$"
    + ', rtol='+str(rtol)
    +", N="+str(N)
    , y=1.1)
    
plt.show()

sz = 6
fig, ax = plt.subplots(figsize=(sz*1.6,sz))

pcm = ax.plot(phis, np.unwrap(np.angle(centreVals)*3)/3, color="#D2419A")
ax.set_xlabel(r"$\phi$")
ax.set_title(r"$\mathrm{Phase}\{P(t)_{" +str(me1)+r","+str(me2)+r"}\}$")


fig.suptitle(r"$P(t)_{"+str(me1)+r","+str(me2)+r"}$"+"\n"
    + r" $V(t) = |"+str(centre)+r"><"+str(centre)+r"|"
    + str(a)+r"\cos("
    + "{:.2f}".format(omega)
    + r"t + \phi"
    + r")$"
    + ', rtol='+str(rtol)
    +", N="+str(N)
    , y=1.1)
    
plt.show()





