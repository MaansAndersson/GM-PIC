import numpy as np
import dace as dc

""" Not for performance """
class Particles(object):
    """ Store the particles in a separate object """

    def __init__(self):
        self.species = None
        self.number_of_particles = None
        self.velocities = None

    @dc.program
    def step():
        for i in dc.map(0:N):
            p += v*dt + x
        
        pass




