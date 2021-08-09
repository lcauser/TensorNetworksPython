from tensornetworks.structures.mps import *
from tensornetworks.structures.mpo import *
from tensornetworks.lattices.mps import spinHalf
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.algorithms.dmrg import dmrg
from tensornetworks.structures.opList import opList
from tensornetworks.structures.gateList import gateList, trotterize, applyGates

N = 100
c = 0.5
s = -1.0

B = tn.tensor((1, 2, 1), [[[np.sqrt(c)]], [[np.sqrt(1-c)]]])
psi = productMPS(2, N, B).orthogonalize(N-1).orthogonalize(0)

ops = opList(spinHalf(), N)
for i in range(N-1):
    ops.add(["n", "x"], [i, i+1], np.sqrt(c*(1-c))*np.exp(-s))
    ops.add(["n", "pu"], [i, i+1], -(1-c))
    ops.add(["n", "pd"], [i, i+1], -c)
ops.add(["x"], [0], np.sqrt(c*(1-c))*np.exp(-s))
ops.add(["pu"], [0], -(1-c))
ops.add(["pd"], [0], -c)

time = 0
for timestep in [1.0, 0.1, 0.01]:
    gates = trotterize(ops, timestep, order=2)
    for k in range(min(int(100/timestep), 1000)):
        applyGates(gates, psi, cutoff=10**-12)
        psi.normalize()
        time += timestep
        print("time = "+str(time)+", maxbonddim = " + str(psi.maxBondDim()))