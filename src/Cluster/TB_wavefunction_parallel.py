##Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import sqrt as sqrt
from numpy import pi as pi
#from numpy.matrix import transpose
from scipy.special import eval_hermite
from scipy.special import factorial
import scipy.optimize
from scipy import stats
from scipy.interpolate import interp1d
from matplotlib import cm
from numpy import sqrt as sqrt
from numpy import pi as pi
import functions
from scipy.sparse import csc_matrix, lil_matrix, save_npz, load_npz
from matplotlib import ticker
from pathlib import Path

# from functions import compute_eigenstate
from functions import *
import generate_lattice
from generate_lattice import generate_sites, generate_octagon, clean_rings
from potential_functions import *
from cont_schrod import hamiltonian, laplace_kernel,index1_to_index2, index2_to_index1
from variation import min_spread
import time
from cycler import cycler
import sys
from joblib import Parallel, delayed

#Load style file
plt.style.use('PaperDoubleFig.mplstyle')

if __name__ == "__main__":

    #Define lattice parameters

    lattice_params = dict()
    lattice_params['length'] = 25.0# 33.0
    lattice_params['depth'] =  float(sys.argv[1])#7.0#float(sys.argv[1])# 1.7#
    lattice_params['cut_off'] = 4.0
    cut_off = lattice_params['cut_off']
    depth = lattice_params['depth']
    oscillator_length = 1 / (2 * pi * (lattice_params['depth'] ** 0.25))
    global_step = 0.03# 1/6  * oscillator_length ##Remember: better to use 1/8! 0.02
    print(global_step)
    lattice_params['global_step'] = global_step
    x_global, y_global = generate_grid((0.0, 0.0), lattice_params['length'] / 2 + lattice_params['cut_off'] +0.5, global_step)
    lattice_params['x_global'] = x_global
    lattice_params['y_global'] = y_global
    lattice_params['n_x'] = len(x_global)

    k0 = 2 * pi
    kx = np.array([1, 0]) * k0
    ky = np.array([0, 1]) * k0
    kp = 1 / sqrt(2) * (kx + ky)  # Quasicrystal diagonal beams
    km = 1 / sqrt(2) * (kx - ky)

    ##QC setup
    delta_y = 121 + np.sqrt(5)  # 0.0#1.9-#0.4
    delta_x = 75.0  # 3+0.216#-1.56-0.36
    # delta_1 = sqrt(7)
    # delta_2 = 323.65
    # delta_3 = np.sqrt(5)
    # delta_4 = 13.2
    phi1 = -2 * pi * delta_x  # 0.34-2*pi*delta_x#0.64536465
    phi2 = -2 * pi * delta_y  # 3.50-2*pi*delta_y#np.sqrt(2)
    phi3 = -2 * pi * 1 / np.sqrt(2) * (delta_x + delta_y)  # -2*pi*1/np.sqrt(2)*(delta_x+delta_y)   # 1.4-2*pi*1/np.sqrt(2)*(delta_x+delta_y)#2*pi/5#+0.5*np.sqrt(2)/2
    phi4 = -2 * pi * 1 / np.sqrt(2) * (delta_x - delta_y)  # 2.7  # 3.2-2*pi*1/np.sqrt(2)*(delta_x-delta_y)#np.sqrt(2)*pi #+0.5qrt(2) * ( delta_x - delta_y)#0.0#   # 3.2-2*pi*1/np.sqrt(2)*(delta_x-delta_y)#np.sqrt(2)*pi #+0.5*np.sqrt(2)/2
    k = np.array([kx, ky, kp, km])
    phis = np.array([phi1, phi2, phi3, phi4])  # phases of the lattice laser beams

    lattice_params['k'] = k
    lattice_params['phis'] = phis

    runtic = time.perf_counter()

    # Create directory for saving files
    # dirname = "/home/emg69/rds/results_3186_sites/results_depth_"+str(depth)
    # dirname = "/rds/user/emg69/hpc-work/results_TB_size_35/results_depth_"+str(depth)
    # dirname = "size_6/results_depth_"+str(lattice_params['depth'])
    dirname = "/rds/user/emg69/hpc-work/QC_optimised/size_"+str(lattice_params['length'])+"/results_depth_"+str(depth)
    # dirname = "test/size_"+str(lattice_params['length'])+"/results_depth_"+str(depth)


    # dirname = "/rds/user/emg69/hpc-work/test/results_depth_"+str(depth)

    Path(dirname).mkdir(parents=True, exist_ok=True)
    print(dirname)
    #
    # plt.subplot(1,1,1, aspect = 'equal')
    # fig = plt.figure(1)
    # plt.subplot(1, 1, 1, aspect='equal')
    # ax =  plt.gca()
    # ax= fig.gca(projection='3d')

    # ax.axis('equal')
    Xglobal, Yglobal = np.meshgrid(x_global, y_global)
    V_global = potential(Xglobal, Yglobal, depth, k, phis)
    # im = plt.pcolor(x_global, y_global, V_global, cmap = 'jet')
    # surf = ax.plot_surface(Xglobal, Yglobal, V_global, cmap='jet',
    #                        linewidth=0, antialiased=False)
    # plt.suptitle("optical potential $V(x,y) \, [E_{rec}]$")
    # plt.xlim(-5.5,5.5)
    # plt.ylim(-5.5,5.5)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    #
    # plt.savefig('potential.png', dpi = 900)
    # # plt.colorbar()
    # plt.show()
    # plt.clf()
    # print('potential saved')
    # fig, ax = plt.subplots(1, 1, dpi=300)
    # ax.axis('equal')
    # # pcm = ax.pcolormesh(x_global, y_global, (result_vec[:,i_state].reshape((n_x_global, n_x_global)).todense()),
    # #                     norm=colors.SymLogNorm(linthresh=1e-3, linscale=0),
    # #                     cmap='RdBu_r')
    # pcm = ax.pcolormesh(x_global, y_global, V_global, cmap='jet')
    # ax.set(xlabel=r'$x (\lambda)$', ylabel='$y (\lambda)$')
    # # ax.axis(xmin=- 2.3, xmax=2.3, ymin=2.3, ymax= 2.3)  # ,option= 'equal')
    # the_divider = make_axes_locatable(ax)
    # color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    # fig.colorbar(pcm, cax=color_axis)  # , extend='both')
    # plt.show()




    # minima = np.load(dirname+'/minima.npy')
    # minima = np.vstack([minima, np.array([0.346, -14.770])])
    # print(minima)

    minima =  generate_sites(x_global[0], x_global[-1], y_global[0], y_global[-1], (0.0, 0.0), lattice_params, mask_radius=lattice_params['length'] / 2)
    # np.load(dirname + '/minima.npy')  #
    # print(minima)
    np.save(dirname + '/phis', phis)
    octagon = generate_octagon(minima, phis)
    ##Look for ring sites in list of minima
    minima_clean, rings_list = clean_rings(minima, octagon)
    np.save(dirname + "/lattice_sites", minima_clean)
    print("lattice sites saved")
    octagon_clean = generate_octagon(minima_clean, phis)
    np.save(dirname + '/octagon', octagon_clean)
    print("octagon sites saved")

    #minima_clean = minima[[0,2,6],:]
    #rings_list = np.array([])
    print(minima_clean)
    np.save(dirname + '/rings_list', rings_list)
    n_rings = rings_list.shape[0]
    lattice_params['rings_list'] = rings_list
    print("CLEANED SITES: " + str(len(minima)))
    print("RINGS: "+ str(rings_list))
    lattice_params['lattice_sites'] = minima_clean
    n_states = minima_clean.shape[0]# + 3*rings_list.shape[0]
    # np.save(dirname + "/lattice_params", lattice_params)
    # minima = np.load("minima.npy")#generate_sites(left, right, down, up, mask_radius=config.p<laquette_length_x / 2)
    print("sites loaded: "+str(n_states)+" sites")
    #fig = plt.figure(1)
    if minima.shape[0]>0:
        x_minima_clean, y_minima_clean = zip(*minima_clean)
        x_minima, y_minima = zip(*minima)

        # ax = fig.add_subplot(1, 1, 1, aspect = 'equal')
        # plt.pcolor(x_global, y_global, V_global, cmap = 'jet', zorder = 1)
        # plt.contourf(x_global, y_global, V_global, 100, cmap = 'jet', zorder = 1)
        plt.scatter(x_minima, y_minima, s = 25, c = 'k', edgecolor = 'k', zorder = 1)
        plt.scatter(x_minima_clean, y_minima_clean, s = 25, c = 'r', edgecolor = 'k', zorder = 3)
    if (rings_list.shape[0])>0:
        x_ring, y_ring = zip(*rings_list)
        plt.scatter(x_ring, y_ring, s = 33, c = 'b', edgecolor = 'k', zorder = 3)

    # for i in range(len(minima)):
    #     site = minima[i]
    #     ax.text(site[0] - 0.1, site[1], str(i), color='black', fontsize=8)
    #
    np.save(dirname+"/minima", minima)
    plt.savefig(dirname+"/lattice_sites_V" + str(depth) + ".png", dpi = 300)
    # plt.show()

    plt.clf()
    sites_number = n_states
    np.save("sites_number", sites_number)

    tic = time.perf_counter()


    print("Number of sites" + str (n_states))
    n_x = lattice_params['n_x']
    n_y = n_x
    window_radius = cut_off + 1.25
    x_window, y_window = generate_grid((0.0, 0.0), window_radius, global_step)
    n_x_window = len(x_window)
    n_y_window = n_x_window
    # try:
    #     wannier_functions_vec = load_npz(dirname+"/wannier_functions.npz")
    #     wannier_functions_vec = wannier_functions_vec.tocsc()

    # # except IOError:
    if Path(dirname + "/wannier_functions.npy").is_file() == True:
        print('Wannier functions already exist ')
        wannier_functions_vec = np.load(dirname + "/wannier_functions.npy")
    if Path(dirname + "/wannier_functions.npy").is_file()==False:
        wannier_functions_vec = lil_matrix((n_x_window * n_y_window, n_states))
        print("Start generation of Wannier Functions")
        #wannier_functions_ring = scipy.sparse.hstack(np.array(
        #    Parallel(n_jobs=1, backend='multiprocessing', batch_size=1, verbose=100)(
        #        delayed(generate_wannier_function)(i_ring, lattice_params, state_number=3, plot=True, ring=True) for i_ring
        #        in range(rings_list.shape[0]))), format="csc")
        wannier_functions_normal = np.hstack(np.array(Parallel(n_jobs=-1, backend='multiprocessing', batch_size=1, verbose=100)(
                delayed(generate_wannier_function)(i_site, lattice_params, plot=False) for i_site in range(minima_clean.shape[0]))))
        #wannier_functions_vec = scipy.sparse.hstack([wannier_functions_normal, wannier_functions_ring], format = "csc")
        wannier_functions_vec = wannier_functions_normal
        print("Wannier functions generated")
        np.save(dirname + "/wannier_functions", wannier_functions_vec)


    ##Compute overlap matrix
    S_matrix = lil_matrix(np.zeros(n_states))
    H_wannier = lil_matrix(np.zeros(n_states))
    if Path(dirname + "/S_matrix.npy").is_file() == True:
        print('H wannier already exist ')
        H_wannier = load_npz(dirname + "/hamiltonian_wannier.npz")
    if Path(dirname + "/hamiltonian_wannier.npz").is_file() == False:
        H_wannier = scipy.sparse.hstack(np.array(
            Parallel(n_jobs=-1, backend='multiprocessing', batch_size=1, verbose=100)(
                delayed(compute_H)(i_site, wannier_functions_vec, minima_clean, cut_off,window_radius, global_step, lattice_params) for i_site in
                range(minima_clean.shape[0]))), format="csc") ##Only compute lower half of H
        H_wannier = (H_wannier + H_wannier.T) #makes H hermitian
        save_npz(dirname+"/hamiltonian_wannier", H_wannier)
        print("H wannier generated")

    if Path(dirname + "/S_matrix.npy").is_file() == True:
        print('S matrix already exist ')
        S_matrix = np.load(dirname + "/S_matrix.npy")
    if Path(dirname + "/S_matrix.npy").is_file()==False:
        S_matrix = scipy.sparse.hstack(np.array(
            Parallel(n_jobs=-1, backend='multiprocessing', batch_size=1, verbose=100)(
                delayed(compute_S)(i_site, wannier_functions_vec, minima_clean, cut_off,window_radius, global_step) for i_site in
                range(minima_clean.shape[0]))), format="csc")
        S_matrix = (S_matrix +S_matrix.T) #makes S symmetric
        S_matrix = S_matrix.todense()
        np.save(dirname+"/S_matrix.npy", S_matrix)
        print("S matrix generated")

    # print(H_wannier)
    ### Lowdin orthogonalization ###
    # save_npz(dirname+"/overlap", S_matrix)
    # non_zero_indices = np.split(S_matrix.indices, S_matrix.indptr)[1:-1]
    # np.fill_diagonal(S_matrix, 0.0)
    # cm = 1/2.54
    # plt.figure(5, figsize = (15*cm, 15*cm))
    # plt.subplot(1,1,1, aspect = 'equal')
    # plt.pcolor(S_matrix, cmap = 'RdBu_r',
    #                 norm=colors.DivergingNorm(vmin=-np.max(np.abs(S_matrix)),
    #                                          vcenter=0., vmax=np.max(np.abs(S_matrix))))
    # plt.colorbar()
    # plt.savefig("S_matrix.png", dpi = 900)
    # plt.clf()
    # S_matrix = np.sort(S_matrix.ravel())
    # site_indices = np.random.randint(0,n_states, 10)
    # for i_state in site_indices:
    #     plt.figure(5)
    #     non_zero_indices= np.argwhere(S_matrix[i_state, :]!=0.0).ravel()
    #     plt.subplot(1,1,1, aspect = 'equal')
    #     # plt.scatter(x_minima_clean, y_minima_clean, s = 12, c = 'grey')
    #     plt.scatter(np.array(x_minima_clean), np.array(y_minima_clean),
    #                 s = 18, c = (S_matrix[i_state,:]), cmap = 'RdBu_r',
    #                 norm = colors.SymLogNorm(1e-12, linscale=1.0, vmin=-1.0, vmax=1.0))
    #
    #                 # norm=colors.DivergingNorm(vmin=-np.max(np.abs(S_matrix)),
    #                                          # vcenter=0., vmax=np.max(np.abs(S_matrix)))
    #
    #     plt.colorbar()
    #     plt.scatter(x_minima_clean[i_state], y_minima_clean[i_state], s = 18, c = 'r')
    #     plt.savefig('overlap_'+str(i_state)+'.png', dpi = 600)
    #     plt.clf()
    #
    S_diag, U_mat = np.linalg.eigh(S_matrix)
    S_sqrt_inv = np.diag(np.sqrt(np.divide(1, S_diag)))
    print(S_diag)
    symm_orthog =S_sqrt_inv.dot(np.conj(U_mat.T))# S_sqrt_inv.dot(np.linalg.inv(U_mat))# S_sqrt_inv.dot(np.conj(U_mat.T))
    symm_orthog =  U_mat.dot(symm_orthog)  # np.dot(U_mat, np.dot(S_sqrt_inv, np.conj(U_mat.T)))
    symm_orthog = csc_matrix(symm_orthog)
    # non_zero_indices = np.split(symm_orthog.indices, symm_orthog.indptr)[1:-1]
    # print(non_zero_indices)
    symm_orthog = symm_orthog.todense()
    # for i_state in site_indices:
    #     plt.figure(5)
    #     non_zero_indices= np.argwhere(symm_orthog[i_state, :]!=0.0).ravel()
    #     plt.subplot(1,1,1, aspect = 'equal')
    #     plt.scatter(x_minima_clean, y_minima_clean, s = 12, c = 'grey')
    #     print(symm_orthog[i_state,:])
    #     plt.scatter(np.array(x_minima_clean), np.array(y_minima_clean), s = 18,
    #                 c = np.array(np.log10(np.abs(symm_orthog[i_state,:]))))
    #     plt.colorbar()
    #     plt.scatter(x_minima_clean[i_state], y_minima_clean[i_state], s = 18, c = 'r')
    #     plt.savefig('symm_orthog_'+str(i_state)+'.png', dpi = 600)
    #     plt.clf()
    # print(symm_orthog)
    print("Symmetric orthogonalization matrix generated")
    H_lowdin = H_wannier.dot(np.conj(symm_orthog.T))
    H_lowdin = csc_matrix(symm_orthog.dot(H_lowdin))
    np.save(dirname+'/hamiltonian_real.npy', H_lowdin)
    print("Hamiltonian saved")
    lowdin_basis_vec = np.zeros((n_x_window*n_y_window, n_states))


    lowdin_basis_vec = np.hstack(np.array(
        Parallel(n_jobs=-1, backend='multiprocessing', batch_size=1, verbose=100)(
            delayed(compute_lowdin)(i_site, wannier_functions_vec, minima_clean, symm_orthog,
                                    window_radius, global_step) for i_site in
            range(minima_clean.shape[0]))))

    # print(lowdin_basis_vec)
    np.save(dirname+'/lowdin_basis_vec', lowdin_basis_vec)
    print('Lowdin basis saved')

    np.save(dirname + "/y_window", y_window)
    np.save(dirname + "/x_window", x_window)

    runtoc = time.perf_counter()
    ##Hubbard U: on-site interaction energy
    hubbard_U = np.power((abs(lowdin_basis_vec)),4)
    hubbard_U = (np.sum(hubbard_U, axis=0) * global_step ** 2)
    print(hubbard_U)
    np.save(dirname + "/hubbard_U", hubbard_U)
    print("hubbard U computed")

