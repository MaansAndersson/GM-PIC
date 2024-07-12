""" Simple skeleton code for a Particle-In-Cell solver """
import numpy as np
import dace as dc

import matplotlib.pyplot as plt
from PIL import Image 

from scipy import sparse
from scipy.sparse import linalg

import os

from FieldSolver import *
            # FEM_3D solver
            # DaCe FDTD solver
            # 3D FFT solver? 
from ParticleHandler import * 



class Simulation(object):
    """ Simulation """
    
    def __init__(self,  )
        """__init__ documentation"""
    
        self.number_of_particles = N # Number of particles per cell

        # Temporary for 1D evaluation to be moved to the
        # grid based solver
        
        # set_up_grid_solver() 
        self.domain_length = 2*np.pi/3.0600
        self.grid_size = N              # Number of grid points 
        self.delta_x   = None           # Discretization size
        
        # Temporary for Time to be moved to the time-integrator
        
        # set_up_time_integrator()
        self.dt        = 0.2
        self.T         = 200*dt

        # Particle initial properties
        V0 = 0.2                        # Beam velocity 
        VT = 0.025                      # Thermal speed
 
        # Particles
        positions 
        velocities

        # Pertubation
        XP1 = 0.1 
        mode = 1
        self.apply_pertubation()

        pass

    def load_from_ceckpoint(self, ):
        
        #set_**()
        pass

    def simulation_step(self, ):
        """ simulation_step """ 

        self.update_particle_positions()
        self.enforce_periodic_boundary_condition()
        self.project_particles_to_grid()
        self.compute_electric_field_potetntial()
        self.project_field_to_particles()
        self.update_particle_velocities()
              
        # Energies
        self.statistics() 
        self.plotter()
         
    def run():   
        for i in range(tsteps):
            self.simulation_step()


def write_to_file():
    pass

def read_from_file():
    pass 



