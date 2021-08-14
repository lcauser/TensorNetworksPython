"""
    Example of TEBD.
    We evolve a biased stochastic East model in real time from its stationary
    state of the unbiased model.
    We measure the norm, and the occupations. When the norm is unchanging,
    we consider it converged.
"""

from tensornetworks.structures.mps import meanfieldMPS
from tensornetworks.lattices.spinHalf import spinHalf
import numpy as np
from tensornetworks.structures.opList import opList
from tensornetworks.algorithms.tebd import tebd, normTEBDobserver, oplistTEBDobserver

N = 100
c = 0.5
s = -1.0
dt = 0.01
tmax = 1000

# Create the initial stationary state
B = [np.sqrt(c), np.sqrt(1-c)]
psi0 = meanfieldMPS(2, N, B).orthogonalize(N-1).orthogonalize(0)

# Create the Hamiltonian
ops = opList(spinHalf(), N)
ops.add(["x"], [0], np.sqrt(c*(1-c))*np.exp(-s))
ops.add(["pu"], [0], -(1-c))
ops.add(["pd"], [0], -c)
for i in range(N-1):
    ops.add(["n", "x"], [i, i+1], np.sqrt(c*(1-c))*np.exp(-s))
    ops.add(["n", "pu"], [i, i+1], -(1-c))
    ops.add(["n", "pd"], [i, i+1], -c)

# Obserer to measure the norm
normObs = normTEBDobserver(10**-10)

# Observer to measure occupations
opObs = opList(spinHalf(), N)
for i in range(N):
    opObs.add("n", i)
opObs = oplistTEBDobserver(opObs)

# Do the evolution
psi = tebd(psi0, ops, tmax, observers=[normObs, opObs], dt=dt)

# Get measurements
norms = normObs.measurements
occs = np.real(opObs.measurements)