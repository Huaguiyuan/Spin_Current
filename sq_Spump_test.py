# Program to calculate the dc spin current across a 2D SQ lattice
# Author: Amin Ahmdi
# Date: May 1, 2017
# Date: June 19, 2017 : test the code compare with conductance steps
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

N_energy = 100
eta = 1.e-7j
mlat = 10                        # width, no spin
nlat = 2                       # length of chain
I_l = np.eye(mlat,dtype=complex) # identity matrix, lattice dimension

sigma_0 = np.array([[1.,0.],
                    [0.,1.]], dtype=complex)
sigma_x = np.array([[0,1.],
                    [1.,0]], dtype=complex)
sigma_y = np.array([[0,-1.j],
                    [1.j,0]], dtype=complex)
sigma_z = np.array([[1.,0.],
                    [0.,-1.]], dtype=complex)
sigma_u = np.array([[0.,1.],
                    [0.,0.]], dtype=complex)
sigma_d = np.array([[0.,0.],
                    [1.,0.]], dtype=complex)

# Constructing sigma-tensor matrix: sig_ut = sigma^+ \otime I
sig_ut = np.kron(sigma_u,I_l)
sig_dt = np.kron(sigma_d,I_l)
sig_zt = np.kron(sigma_z,I_l)




def make_h_tau(mlat, nlat, imp_dens=0.0, imp_amp=0.0):
    """ Constructs the Hamiltonian and the connection 
    matrix of 1D spin depndent chain, 
    mlat: 2 spin-states,
    nlat: length of chain,
    imp_dens: density of impurities
    imp_amp: amplitude of the scatterer
    """
    tau = -np.eye(2*mlat,dtype=complex)
    h = np.zeros((nlat+1,2*mlat,2*mlat), dtype=complex)
    N_imp = int(nlat*imp_dens)               # number of sites with impurities
    imp_arr = nr.randint(1,nlat, size=N_imp) # impurity sites are chosen

    # magnetic site sigma_z, \Omega_\| = 1
    for i in range(2*mlat):
        h[0][i][i] = -1                      # to test magnetic site is off
        
    # latice hamiltonian i=1 to i=N
    # this must be changed for actual spin current calculation
    # since in the lattice spin's site are not connected
    
    for i in range(1,nlat):
        for j in range(2*mlat-1):
            h[i][j,j+1] = -1.0
            h[i][j+1,j] = np.conjugate(h[i][j,j+1])

    # magnetic on-site impurities
    for im_site in imp_arr:
        h[im_site] += [[0.0, imp_amp],
                      [imp_amp,0.0]]
    
    return h, tau



def g_lead_dec(E, tau, h): 
    """ Compute the lead's Green's function using decimation
    method. 
    
    Inputs:
    E: energy
    tau: connection matrix between slices in the lead
    h: The Hamiltonian of one slice in the lead
    
    Return:
    Lead's Green's function
    """
    eta = 1.e-7j               # infinitesimal imaginary for retarded G.Fs.
    
    ee = E + eta
    # initialize alpha, beta, eps, eps_s
    alpha = tau
    beta = np.conjugate(tau.T)
    eps = h[1]
    eps_s = h[1]

    for i_dec in range(30):
        aux = lg.inv(ee*np.kron(sigma_0,I_l) - eps)
        aux1 = np.dot(alpha,np.dot(aux,beta))
        eps_s = eps_s + aux1
        eps = eps + aux1
        aux1 = np.dot(beta, np.dot(aux, alpha))
        eps = eps + aux1
        alpha = np.dot(alpha,np.dot(aux, alpha))
        beta = np.dot(beta, np.dot(aux, beta))


    g_l = lg.inv(ee*np.kron(sigma_0,I_l) - eps_s)
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
# physical dimension is given, do not consider the spin
h, tau = make_h_tau(mlat, nlat, 0.0, 0.0)

# to save right-to-left Green's function
g_right = np.zeros((nlat+1,2*mlat,2*mlat), dtype=complex)

# loop over energy 
for ie in range(N_energy):

    # imaginary part from the Green's function
    # of lead is transferred, no need for extra imaginary
    energy = -2.5 +  ie*(5.0/N_energy)

    # The lead's Green's function
    g_l = g_lead_dec(energy, tau, h)
    # Instead of decimation the Lead Green's function is given
    #g_l = g_lead(energy) * np.kron(sigma_0,I_l)
    

    # The self-energy due to the right reservoir
    sigma_r = np.dot(tau, np.dot(g_l,tau))
    sigma_l = sigma_r
    sigma_r_dg = np.conjugate(sigma_r.T)

    # the Green's function of the last site "nlat"
    
    # G_n = lg.inv( energy*np.kron(sigma_0,I_l) - h[nlat] - sigma_r)
    # g_right[nlat] = G_n
    # sigma_n = np.dot(tau, np.dot(G_n, tau))

    # # Right-to-Left Sweep 
    # for i_len in range(nlat-1,0,-1):
    #     G_n = lg.inv( energy*np.kron(sigma_0,I_l) - h[i_len] - sigma_n )
    #     g_right[i_len] = G_n
    #     sigma_n = np.dot(tau, np.dot(G_n, tau))

    # Full surface Green's function at the magnetic site G_{00}
    
    # G_n = lg.inv( energy*np.kron(sigma_0,I_l) - h[0] - sigma_n)

    # G_ln = G_n
    # G_rn = G_n
    # # Left-to-Right Sweep
    # for i_len in range(1,nlat+1):
    #     G_ln = np.dot(G_ln,np.dot(tau,g_right[i_len]))
    #     G_rn = np.dot(np.dot(g_right[i_len], tau), G_rn)

    # #  at final step  G_{N0} & G_{0N} are calculated

    # G_ln_a = np.conjugate(G_ln.T)
    # G_rn_a = np.conjugate(G_rn.T)


    # aux1 = np.dot(G_rn, np.dot(sig_ut, G_ln))
    # aux2 = np.dot(G_rn_a, np.dot(sig_dt, G_ln_a))
    # aux = np.dot(aux1,np.dot(sig_zt,aux2))

    # j_s = rho_lead(energy)**(mlat) * np.trace(aux)
    # #j_s = np.trace(aux)
    
    # print " %.3f   %.3f  "  %(energy.real, abs(j_s))


    ### To test the code for quantum conductance ###

    # total Green's function, one site between leads
    G_d = lg.inv( energy*np.kron(sigma_0,I_l) - h[1] -
                  sigma_r  - sigma_l) 
    G_d_dg = np.conjugate(G_d.T)

    gamma_l = -1j * (sigma_r - sigma_r_dg)
    gamma_r = np.conjugate(gamma_l.T)

    # G * gamma_l * G_dg * gamma_r
    auxg = np.dot(G_d, np.dot(gamma_l, np.dot(G_d_dg, gamma_r)))

    gg = np.trace(auxg)

    print energy.real, gg.real

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
