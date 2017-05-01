# Program to calculate the dc spin current across a 2D SQ lattice
# Author: Amin Ahmdi
# Date: May 1, 2017
# ################################################################
# 
# To consider the spin in this problem, we make a copy of each row:
# 
# 0--o--o--o--o--o--o--o--o ... o--o--o--o--a
# 0--o--o--o--o--o--o--o--o ... o--o--o--o--a
# 
#
##################################################################
import numpy as np
import numpy.linalg as lg
import numpy.random as nr

N_energy = 200
eta = 1.e-7j
mlat = 10                        # two rows to mimic spin state
nlat = 200
I = np.eye(mlat,dtype=complex)

def make_h_tau(mlat, nlat, imp_dens=0.0, imp_amp=0.0):
    """ Constructs the Hamiltonian and the connection 
    matrix of 1D spin depndent chain, 
    mlat: 2 spin-states,
    nlat: length of chain,
    imp_dens: density of impurities
    imp_amp: amplitude of the scatterer
    """
    tau = -np.eye(mlat,dtype=complex)
    h = np.zeros((nlat+1,mlat,mlat), dtype=complex)
    N_imp = int(nlat*imp_dens)               # number of sites with impurities
    imp_arr = nr.randint(1,nlat, size=N_imp) # impurity sites are chosen

    # magnetic site sigma_z, \Omega_\| = 1
    for i in range(mlat):
        h[0][i][i] = (-1)**i
        
    for im_site in imp_arr:
        h[im_site] = [[0.0, imp_amp],
                      [imp_amp,0.0]]
    
    return h, tau



def g_lead_dec(E, tau, h):
    """ Compute the lead's Green's function using decimation
    method. Inputs:
    E: energy
    tau: connection matrix between slices
    h: The Hamiltonian of the slice
    Return:
    Lead's Green's function
    """
    eta = 1.e-7j               # tiny imaginary for retarded G.Fs.
    I = np.eye(mlat,dtype=complex)
    
    ee = E + eta
    # initialize alpha, beta, eps, eps_s
    alpha = tau
    beta = np.conjugate(tau.T)
    eps = h
    eps_s = h

    for i_dec in range(30):
        aux = lg.inv(ee*I - eps)
        aux1 = np.dot(alpha,np.dot(aux,beta))
        eps_s = eps_s + aux1
        eps = eps + aux1
        aux1 = np.dot(beta, np.dot(aux, alpha))
        eps = eps + aux1
        alpha = np.dot(alpha,np.dot(aux, alpha))
        beta = np.dot(beta, np.dot(aux, beta))


    g_l = lg.inv(ee*I - eps_s)
    return g_l

def g_lead(E):
    """ The lead's Green's function is given as a analytical form 
    for the square lattice
    """
    if (np.abs(E)<2):
        g_l = E.real/2. - (1.0j)*np.sqrt(1 - E.real**2/4.0)
    else:
        g_l = E.real/2. * ( 1. - np.sqrt(1 - 4.0/E.real**2) )

    return g_l

def rho_lead(E):
    """ The density of state in a semi-infinte chain"""
    if (np.abs(E)<2):
        rho_l = np.sqrt(1.0 - E.real**2/4.0)
    else:
        rho_l = 0.
        
    return rho_l

# construct the Hamiltonian
h, tau = make_h_tau(mlat, nlat, 0.0, 0.0)

# to save right-to-left Green's function
g_right = np.zeros((nlat+1,mlat,mlat), dtype=complex)

# loop over energy 
for ie in range(N_energy):

    energy = -4.0 +  ie*(8.0/N_energy) + 1.0j*eta

    # The lead's Green's function
    # g_l = g_lead_dec(energy, tau, h)
    # Instead of decimation the Lead Green's function is given
    g_l = g_lead(energy)*I
    

    # The self-energy due to the right reservoir
    sigma_r = np.dot(tau, np.dot(g_l,tau))
    sigma_r_dg = np.conjugate(sigma_r.T)

    # the Green's function of the last site "nlat"
    
    G_n = lg.inv(energy*I - h[nlat] - sigma_r)
    g_right[nlat] = G_n
    sigma_n = np.dot(tau, np.dot(G_n, tau))

    # Right-to-Left Sweep 
    for i_len in range(nlat-1,0,-1):
        G_n = lg.inv(energy*I - h[i_len] - sigma_n)
        g_right[i_len] = G_n
        sigma_n = np.dot(tau, np.dot(G_n, tau))

    # Full surface Green's function at the magnetic site G_{00}
    
    G_n = lg.inv(energy*I - h[0] - sigma_n)

    G_ln = G_n
    G_rn = G_n
    # Left-to-Right Sweep
    for i_len in range(1,nlat+1):
        G_ln = np.dot(G_ln,np.dot(tau,g_right[i_len]))
        G_rn = np.dot(np.dot(g_right[i_len], tau), G_rn)

    #  at final step  G_{N0} is calculated

    aux = (abs(G_rn[0,0])*abs(G_ln[1,1]))**2 - \
          (abs(G_rn[1,0])*abs(G_ln[0,1]))**2

    j_s = rho_lead(energy)**2 * aux
    
    print " %.3f   %.3f  %.3f  %.3f   %.3f   %.3f   %.3f " \
        %(energy.real, abs(G_rn[0,0]), abs(G_ln[1,1]), abs(G_rn[1,0]),
          abs(G_ln[0,1]), abs(G_rn[0,0]) * abs(G_ln[1,1]), j_s)


# Result of conductance no vertical connection :
  # 2.5 +-+------+-------+--------+--------+-------+--------+-------+------+-+   
  #     +        +       +        +        +       +        +       +        +   
  #     |                                          "data.dat" us 1:2 ******* |   
  #     |                                                                    |   
  #   2 +-+               **********************************               +-+   
  #     |                 *                                *                 |   
  #     |                 *                                 *                |   
  #     |                *                                  *                |   
  # 1.5 +-+              *                                  *              +-+   
  #     |                *                                  *                |   
  #     |                *                                  *                |   
  #   1 +-+              *                                  *              +-+   
  #     |                *                                  *                |   
  #     |                *                                  *                |   
  #     |                *                                  *                |   
  # 0.5 +-+              *                                  *              +-+   
  #     |                *                                  *                |   
  #     |                *                                  *                |   
  #     +        +       *        +        +       +        *       +        +   
  #   0 ******************--------+--------+-------+--------******************   
  #    -4       -3      -2       -1        0       1        2       3        4   
