"""
    Example of DMRG on a biased stochastic East model.
    The Hamiltonian must be constructed as an efficent MPO.
    We show a random MPS, or mean field MPS as an inital guess.
    Afterwards, we measure occupations and NN correlations.
"""

from tensornetworks.structures.mps import *
from tensornetworks.structures.mpo import *
from tensornetworks.lattices.spinHalf import spinHalf
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.algorithms.dmrg import dmrg
from tensornetworks.structures.opList import opList, opExpectation

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
B = [np.sqrt(c), np.sqrt(1-c)]
psi = meanfieldMPS(2, N, B)
#psi = randomMPS(2, N, 1)

# Do DMRG
psi2 = dmrg(H, psi, nsites=2)

# Measure occupations
ops = opList(sh, N)
for i in range(N):
    ops.add("pu", i)
occupations = opExpectation(ops, psi2)

# Measure NN correlations
ops = opList(sh, N)
for i in range(N-1):
    ops.add(["pu", "pu"], [i, i+1])
correlations = opExpectation(ops, psi2)