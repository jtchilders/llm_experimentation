import os
local_rank = int(os.getenv('PMI_LOCAL_RANK', 0))
rank = int(os.getenv('PMI_RANK', 0))
world_size = int(os.getenv('PMI_SIZE', 1))

from mpi4py import MPI
COMM_WORLD = MPI.COMM_WORLD
