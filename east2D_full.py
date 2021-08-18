from tensornetworks.structures.peps import *
from tensornetworks.structures.opList2d import opList2d
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.lattices.spinHalf import spinHalf
from tensornetworks.structures.environment import *
from tensornetworks.algorithms.tpeps import *
import copy

# Parameters
c = 0.5
s = 1.0
N = 10
chi = 100

# Create the spinset
sh = spinHalf()

# Create the initial state
states = []
for i in range(N):
    states2 = []
    for j in range(N):
        states2.append("dn")
    states.append(states2)
states[0][0] = "up"
states[N-1][N-1] = "s"
psi0 = productPEPS(sh, states)
psi = copy.deepcopy(psi0)

# Create the gate
gate = np.exp(-s)*np.sqrt(c*(1-c))*contract(np.reshape(sh.op("pu"), (2, 2, 1)), np.reshape(sh.op("x"), (2, 2, 1)), 2, 2)
gate -= (1-c) * contract(np.reshape(sh.op("pu"), (2, 2, 1)), np.reshape(sh.op("pu"), (2, 2, 1)), 2, 2)
gate -= (c) * contract(np.reshape(sh.op("pu"), (2, 2, 1)), np.reshape(sh.op("pd"), (2, 2, 1)), 2, 2)


# Perform tPEPS
Ds = [1, 2, 3]
dts = [0.5, 0.1, 0.05, 0.01]
maxiter = 100

for D in Ds:
    if D > 1:
        dts = [0.01]
        maxiter = 1000
    for dt in dts:
        psi = tpeps(psi, gate, dt, D, maxiter, chi)

    
# Measure occupations
env = environment(psi, chi)
opList = opList2d(sh, N)
for i in range(N):
    for j in range(N):
        opList.add(["n"], [i, j], 0)
occs = reshape(np.real(expectationOpListEnv(env, opList)), (N, N))


# Measure energy
opList = opList2d(sh, N)
for i in range(N):
    for j in range(N):
        if j != N-1:
            opList.add(["n", "x"], [i, j], 0, np.exp(-s)*np.sqrt(c*(1-c)))
            opList.add(["n", "pu"], [i, j], 0, -(1-c))
            opList.add(["n", "pd"], [i, j], 0, -c)
        if i != N-1:
            opList.add(["n", "x"], [i, j], 1, np.exp(-s)*np.sqrt(c*(1-c)))
            opList.add(["n", "pu"], [i, j], 1, -(1-c))
            opList.add(["n", "pd"], [i, j], 1, -c)
            
energy = 0
for i in range(len(opList.ops)):
    psi2 = copy.deepcopy(psi)
    ops = opList.ops[i]
    sites = opList.sites[i]
    coeff = opList.coeffs[i]
    direct = opList.directions[i]
    
    for j in range(len(ops)):
        A = psi2.tensors[sites[0]+direct*j][sites[1]+(1-direct)*j]
        O = sh.op(ops[j])
        A = contract(A, O, 4, 1)
        psi2.tensors[sites[0]+direct*j][sites[1]+(1-direct)*j] = A
    
    energy += coeff * dotPEPS(psi, psi2)


                