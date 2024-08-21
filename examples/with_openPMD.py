from mpi4py import MPI
import openpmd_api as io 

# Temp

series = io.Series(
        "/home/mansande/Develop/Plasma-PEPSC/openPMD-example-datasets/example-2d/hdf5/data%T.h5",
        io.Access.read_only)

for k_i, i in series.iterations.items():
    print("Iteration: {0}".format(k_i))

    for k_m, m in i.meshes.items():
        print("  Mesh '{0}' attributes:".format(k_m))
        for a in m.attributes:
            print("    {0}".format(a))

    for k_p, p in i.particles.items():
        print("  Particle species '{0}' attributes:".format(k_p))
        for a in p.attributes:
            print("    {0}".format(a))
