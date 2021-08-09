from tensornetworks.structures.mps import *
from tensornetworks.structures.mpo import *
from tensornetworks.lattices.mps import spinHalf
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.algorithms.dmrg import dmrg
from tensornetworks.structures.opList import opList
from tensornetworks.structures.gateList import gateList, trotterize, applyGates

N = 20
c = 0.5
s = -1.0
timestep = 0.01

ops = opList(spinHalf(), 20)
for i in range(N-1):
    ops.add(["n", "x"], [i, i+1], np.sqrt(c*(1-c))*np.exp(-s))
    ops.add(["n", "pu"], [i, i+1], -(1-c))
    ops.add(["n", "pd"], [i, i+1], -c)
ops.add(["x"], [0], np.sqrt(c*(1-c))*np.exp(-s))
ops.add(["pu"], [0], -(1-c))
ops.add(["pd"], [0], -c)
gates = trotterize(ops, timestep, order=2)

B = tn.tensor((1, 2, 1), [[[np.sqrt(c)]], [[np.sqrt(1-c)]]])
#psi = productMPS(2, N, B).orthogonalize(N-1).orthogonalize(0)


for k in range(1000):
    applyGates(gates, psi, cutoff=10**-12)
    psi.normalize()
    print("maxbonddim = " + str(psi.maxBondDim()))