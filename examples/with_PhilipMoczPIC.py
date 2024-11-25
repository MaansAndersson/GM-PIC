import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

"""
Create Your Own Plasma PIC Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the 1D Two-Stream Instability
Code calculates the motions of electron under the Poisson-Maxwell equation
using the Particle-In-Cell (PIC) method

from https://github.com/pmocz/pic-python

Adapted to GMM

"""

import GM_PIC

def getAcc( pos, Nx, boxsize, n0, Gmtx, Lmtx ):
    """
    Calculate the acceleration on each particle due to electric field
    pos      is an Nx1 matrix of particle positions
    Nx       is the number of mesh cells
    boxsize  is the domain [0,boxsize]
    n0       is the electron number density
    Gmtx     is an Nx x Nx matrix for calculating the gradient on the grid
    Lmtx     is an Nx x Nx matrix for calculating the laplacian on the grid
    a        is an Nx1 matrix of accelerations
    """
    # Calculate Electron Number Density on the Mesh by 
    # placing particles into the 2 nearest bins (j & j+1, with proper weights)
    # and normalizing
    N          = pos.shape[0]
    dx         = boxsize / Nx
    j          = np.floor(pos/dx).astype(int)
    jp1        = j+1
    weight_j   = ( jp1*dx - pos  )/dx
    weight_jp1 = ( pos    - j*dx )/dx
    jp1        = np.mod(jp1, Nx)   # periodic BC
    n  = np.bincount(j[:,0],   weights=weight_j[:,0],   minlength=Nx);
    n += np.bincount(jp1[:,0], weights=weight_jp1[:,0], minlength=Nx);
    n *= n0 * boxsize / N / dx
    
    # Solve Poisson's Equation: laplacian(phi) = n-n0
    phi_grid = spsolve(Lmtx, n-n0, permc_spec="MMD_AT_PLUS_A")
    
    # Apply Derivative to get the Electric field
    E_grid = - Gmtx @ phi_grid
    
    # Interpolate grid value onto particle locations
    E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]
    
    a = -E
    
    return a


def main():
    """ Plasma PIC simulation """

    compress = int(input('compress? 1/0: '));
    
    # Simulation parameters
    N         = 40000   # Number of particles
    Nx        = 400     # Number of mesh cells
    t         = 0       # current time of the simulation
    tEnd      = 50      # time at which simulation ends
    dt        = 1       # timestep
    boxsize   = 50      # periodic domain [0,boxsize]
    n0        = 1       # electron number density
    vb        = 3       # beam velocity
    vth       = 1       # beam width
    A         = 0.1     # perturbation
    plotRealTime = True # switch on for plotting as the simulation goes along
    
    # Generate Initial Conditions
    np.random.seed(5)            # set the random number generator seed
    # construct 2 opposite-moving Guassian beams
    pos  = np.random.rand(N,1) * boxsize
    vel  = vth * np.random.randn(N,1) + vb
    Nh = int(N/2)
    vel[Nh:] *= -1
    # add perturbation
    vel *= (1 + A*np.sin(2*np.pi*pos/boxsize))

    # Construct matrix G to computer Gradient  (1st derivative)
    dx = boxsize/Nx
    e = np.ones(Nx)
    diags = np.array([-1,1])
    vals  = np.vstack((-e,e))
    Gmtx = sp.spdiags(vals, diags, Nx, Nx);
    Gmtx = sp.lil_matrix(Gmtx)
    Gmtx[0,Nx-1] = -1
    Gmtx[Nx-1,0] = 1
    Gmtx /= (2*dx)
    Gmtx = sp.csr_matrix(Gmtx)
    
    # Construct matrix L to computer Laplacian (2nd derivative)
    diags = np.array([-1,0,1])
    vals  = np.vstack((e,-2*e,e))
    Lmtx = sp.spdiags(vals, diags, Nx, Nx);
    Lmtx = sp.lil_matrix(Lmtx)
    Lmtx[0,Nx-1] = 1
    Lmtx[Nx-1,0] = 1
    Lmtx /= dx**2
    Lmtx = sp.csr_matrix(Lmtx)
    
    # calculate initial gravitational accelerations
    acc = getAcc( pos, Nx, boxsize, n0, Gmtx, Lmtx )
    
    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    
    # prep figure
    fig = plt.figure(figsize=(5,4), dpi=80)

    # store init vel
    vel0 = vel.copy()
    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        
        vel += acc * dt/2.0
        
        # drift (and apply periodic boundary conditions)
        pos += vel * dt
        pos = np.mod(pos, boxsize)
        
        # update accelerations
        acc = getAcc( pos, Nx, boxsize, n0, Gmtx, Lmtx )
        

        # (1/2) kick
        vel += acc * dt/2.0
        
        # update time
        t += dt
        
        #print(np.sum(np.square(vel)))
        # plot in real time - color 1/2 particles blue, other half red
        if plotRealTime or (i == Nt-1):
            plt.cla()
            plt.scatter(pos[0:Nh],vel[0:Nh],s=.4,color='blue', alpha=0.5)
            plt.scatter(pos[Nh:], vel[Nh:], s=.4,color='red',  alpha=0.5)
            plt.axis([0,boxsize,-6,6])
            plt.pause(0.001)
        
        if (i == round(0.3*Nt) and compress == 1):
            pos_t = 0.*pos.copy()
            vel_t = 0.*vel.copy()
            total_elemets = int(Nx/1)
            mean_V = 0
            mean__pV = (np.mean(vel[:]))
            for element in range(0,total_elemets):
                print('Training and evaluating elemnet: ',element)
                local_idx_ic1 = ((pos[:] < (element+1)*boxsize/total_elemets) \
                        * (pos[:] > (element)*boxsize/total_elemets) \
                        * (vel0[:] != 0)).nonzero()
                
                print('From: ',element*boxsize/total_elemets)
                print('To: ',(element+1)*boxsize/total_elemets)

                number_components = 4
                data_ic1 = np.array([vel[local_idx_ic1].copy()]).T
                model_ic1 = GM_PIC.GaussianMixtureModel(nr_of_components = number_components,
                                                    data  = data_ic1.copy())
                # If position is an independent Gaussian (it's not)
                #modelP1 = GM_PIC.GaussianMixtureModel(nr_of_components = number_components,
                #                                    data  = data_p1)
                #modelP2 = GM_PIC.GaussianMixtureModel(nr_of_components = number_components,
                #                                    data  = data_p2)
                #modelP2.train(400)
                #modelP1.train(400)
                

                #model.initial_guess()
                #model.restart?
                print('---')
                #model_ic1.inspect()
                model_ic1.train(nr_of_steps = 100)

                V_ic1 = model_ic1.evaluate_weighted(len(vel[local_idx_ic1])).T
                mean_V_ic1 = model_ic1.get_mean().T
                alpha_V_ic1 = model_ic1.pi_
                P_ic1 = np.random.uniform(element*boxsize/total_elemets,(element+1)*boxsize/total_elemets,V_ic1.shape)
                
                mean_V += np.dot(mean_V_ic1,alpha_V_ic1)
                
                pos_t[local_idx_ic1] = P_ic1[:]
                vel_t[local_idx_ic1] = V_ic1[:]
                
                fig = plt.figure(2, figsize=(5,4), dpi=80)
                plt.scatter(x=P_ic1, y=V_ic1, s=0.4, color='blue', alpha = 0.5)#, linewidths=1)
                plt.axis([0,boxsize,-6,6])

                #plt.scatter(x=mean_P_ic1, y=mean_V_ic1, s=4,color='orange')#, linewidths=1)
                #plt.scatter(x=mean_P_ic2, y=mean_V_ic2, s=4,color='k')#, linewidths=1)
                
                #plt.show()
                plt.pause(1e-6)
            print(mean_V)
            print(mean__pV)
            pos[:] = pos_t[:]
            vel[:] = vel_t[:]
            input()

        if (i == round(0.3*Nt) and compress > 1):
            pos_t = 0.*pos.copy()
            vel_t = 0.*vel.copy()
            total_elemets = int(Nx/10)
            mean_V = 0
            mean__pV = (np.mean(vel[:]))
            for element in range(0,total_elemets):
                print('Training and evaluating elemnet: ',element)
                local_idx_ic1 = ((pos[:] < (element+1)*boxsize/total_elemets) \
                        * (pos[:] > (element)*boxsize/total_elemets) \
                        * (vel0[:] > 0)).nonzero()
                
                local_idx_ic2 = ((pos[:] < (element+1)*boxsize/total_elemets) \
                        * (pos[:] > (element)*boxsize/total_elemets) \
                        * (vel0[:] < 0)).nonzero()

                print('From: ',element*boxsize/total_elemets)
                print('To: ',(element+1)*boxsize/total_elemets)

                number_components = 4
                data_ic1 = np.array([vel[local_idx_ic1].copy()]).T
                data_ic2 = np.array([vel[local_idx_ic2].copy()]).T
                data_p1 = np.array([pos[local_idx_ic1].copy()]).T
                data_p2 = np.array([pos[local_idx_ic2].copy()]).T
                model_ic1 = GM_PIC.GaussianMixtureModel(nr_of_components = number_components,
                                                    data  = data_ic1.copy())
                model_ic2 = GM_PIC.GaussianMixtureModel(nr_of_components = number_components,
                                                    data  = data_ic2.copy())
                
                # If position is an independent Gaussian (it's not)
                #modelP1 = GM_PIC.GaussianMixtureModel(nr_of_components = number_components,
                #                                    data  = data_p1)
                #modelP2 = GM_PIC.GaussianMixtureModel(nr_of_components = number_components,
                #                                    data  = data_p2)
                #modelP2.train(400)
                #modelP1.train(400)
                

                #model.initial_guess()
                #model.restart?
                print('---')
                #model_ic1.inspect()
                model_ic1.train(nr_of_steps = 100)
                #model_ic1.inspect()
                print('---')
                #model_ic2.inspect()
                model_ic2.train(nr_of_steps = 100)
                #model_ic2.inspect()

                V_ic1 = model_ic1.evaluate_weighted(len(vel[local_idx_ic1])).T
                mean_V_ic1 = model_ic1.get_mean().T
                alpha_V_ic1 = model_ic1.pi_
                V_ic2 = model_ic2.evaluate_weighted(len(vel[local_idx_ic2])).T
                mean_V_ic2 = model_ic2.get_mean().T
                alpha_V_ic2 = model_ic2.pi_
                
                P_ic1 = np.random.uniform(element*boxsize/total_elemets,(element+1)*boxsize/total_elemets,V_ic1.shape)
                P_ic2 = np.random.uniform(element*boxsize/total_elemets,(element+1)*boxsize/total_elemets,V_ic2.shape)
                #P_ic1 = modelP1.evaluate_weighted(len(pos[local_idx_ic1])).T
                #P_ic2 = modelP2.evaluate_weighted(len(pos[local_idx_ic2])).T

                mean_V += np.dot(mean_V_ic1,alpha_V_ic1) + np.dot(mean_V_ic2,alpha_V_ic2)
                
                #print(mean(mean_V_ic1)+sum(sum(mean_V_ic2))/len(mean_V_ic2))
                #print(sum(sum(vel[:]))/len(v[:]))
                pos_t[local_idx_ic1] = P_ic1[:]
                vel_t[local_idx_ic1] = V_ic1[:]
                pos_t[local_idx_ic2] = P_ic2[:]
                vel_t[local_idx_ic2] = V_ic2[:]
                
                fig = plt.figure(2, figsize=(5,4), dpi=80)
                plt.scatter(x=P_ic1, y=V_ic1, s=0.4, color='blue', alpha = 0.5)#, linewidths=1)
                plt.scatter(x=P_ic2, y=V_ic2, s=0.4, color='red', alpha = 0.5)#, linewidths=1)
                plt.axis([0,boxsize,-6,6])

                #plt.scatter(x=mean_P_ic1, y=mean_V_ic1, s=4,color='orange')#, linewidths=1)
                #plt.scatter(x=mean_P_ic2, y=mean_V_ic2, s=4,color='k')#, linewidths=1)
                
                #plt.show()
                plt.pause(1e-6)
            print(mean_V)
            print(mean__pV)
            pos[:] = pos_t[:]
            vel[:] = vel_t[:]
            input()

        #plt.figure(2+element)
                      # Save figure
    plt.xlabel('x')
    plt.ylabel('v')
    plt.savefig('pic.png',dpi=240)
    plt.show()
    return 0


if __name__== "__main__":
  main()
