from tensornetworks.structures.mps import *
from tensornetworks.structures.mpo import *
from tensornetworks.lattices.mps import spinHalf
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.algorithms.dmrg import dmrg

# Parameters
c = 0.5
s = -1.0
N = 100

# Load spin half
sh = spinHalf()

# Make the Hamiltonian
A = -np.exp(-s)*np.sqrt(c*(1-c))*sh.op("x") + c*sh.op("pd") + (1-c)*sh.op("pu")
M = tn.tensor((3, 2, 2, 3))
M[0, :, :, 0] = sh.op("id")
M[2, :, :, 2] = sh.op("id")
M[1, :, :, 0] = A
M[2, :, :, 1] = sh.op("pu")
M1 = copy.deepcopy(M[2:3, :, :, :])
M1[0, :, :, 0] = A
H = uniformMPO(2, N, M)
H.tensors[0] = M1

# Make equilibrium MPS
B = tn.tensor((1, 2, 1), [[[np.sqrt(c)]], [[np.sqrt(1-c)]]])
#B = tn.tensor((1, 2, 1), [[[0]], [[1]]])
#psi = productMPS(2, N, B)
#psi = randomMPS(2, N, 1)

psi2 = dmrg(H, psi)