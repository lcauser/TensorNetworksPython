from tensornetworks.structures.peps import *
from tensornetworks.structures.opList2d import opList2d
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.lattices.spinHalf import spinHalf
from tensornetworks.structures.environment import *
from tensornetworks.algorithms.tpeps import *
import copy


def calculateEnergy(psi, chi):
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
        
        energy += coeff * dotPEPS(psi, psi2, chi)
    
    return np.real(energy)

# Parameters
c = 0.5
s = 0.1
N = 6
chi = 20
maxiter = 1000

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

# Create the gate
gate = np.exp(-s)*np.sqrt(c*(1-c))*contract(np.reshape(sh.op("pu"), (2, 2, 1)), np.reshape(sh.op("x"), (2, 2, 1)), 2, 2)
gate -= (1-c) * contract(np.reshape(sh.op("pu"), (2, 2, 1)), np.reshape(sh.op("pu"), (2, 2, 1)), 2, 2)
gate -= (c) * contract(np.reshape(sh.op("pu"), (2, 2, 1)), np.reshape(sh.op("pd"), (2, 2, 1)), 2, 2)


# Perform tPEPS
Ds = []
dts = [1.0, 0.1, 0.05]
dt = 0.01
chis = []
energies = []
Zs = []
D = 1
chi = 10
energy = calculateEnergy(psi0, 1)

#%% D = 1

# Quickly get closer to the correct solution
for dt in dts:
    psi0, Z = tpeps(psi0, gate, dt, 1, maxiter, chi)
psi = psi0

"""
lastEnergy = -10
converge = False
while not converge:
    psi, Z = tpeps(psi, gate, dt, 1, maxiter, 1, tol=0)
    energy = Z
    if energy - lastEnergy < 10**-4:
        converge = True
    lastEnergy = energy
    energies.append(energy)
"""

dt = 0.01

psi, Z = tpeps(psi, gate, dt, 1, maxiter, 1, tol=10**-7)
energy = Z
energies.append(energy)

psi, Z = tpeps(psi, gate, dt, 2, maxiter, 8, tol=10**-7)
energy = Z
energies.append(energy)

psi, Z = tpeps(psi, gate, dt, 3, maxiter, 18, tol=10**-7)
energy = Z
energies.append(energy)

#%% D = 2
"""
# Do a dt = 0.01 to get closer to solution
dt = 0.01
lastEnergy = -10
converge = False
while not converge:
    psi, Z = tpeps(psi, gate, dt, 2, maxiter, 20, tol=0)
    energy = Z
    if energy - lastEnergy < 10**-2:
        converge = True
    lastEnergy = energy
psi0 = copy.deepcopy(psi)


# Do for dt = 0.001, incrasing chi along the way
dt = 0.01
chis = [8]
#psi = copy.deepcopy(psi0)
for chi in chis:
    lastEnergy = -10
    converge = False
    while not converge:
        psi, Z = tpeps(psi, gate, dt, 2, maxiter, chi, tol=0)
        energy = Z
        if energy - lastEnergy < 10**-4:
            converge = True
        lastEnergy = energy
    energies.append(energy)

#%% D = 3

# Do a dt = 0.01 to get closer to solution
dt = 0.01
lastEnergy = -10
converge = False
while not converge:
    psi, Z = tpeps(psi, gate, dt, 3, maxiter, 20, tol=0)
    energy = Z
    if energy - lastEnergy < 10**-2:
        converge = True
    lastEnergy = energy


# Do for dt = 0.001, incrasing chi along the way
dt = 0.01
chis = [18]
for chi in chis:
    lastEnergy = -10
    converge = False
    while not converge:
        psi, Z = tpeps(psi, gate, dt, 3, maxiter, chi, tol=0)
        energy = Z
        if energy - lastEnergy < 10**-4:
            converge = True
        lastEnergy = energy
    energies.append(energy)
"""