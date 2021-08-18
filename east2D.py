from tensornetworks.structures.peps import *
from tensornetworks.structures.opList2d import opList2d
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.lattices.spinHalf import spinHalf
from tensornetworks.structures.environment import *
import copy

# Parameters
c = 0.5
s = -1.0
N = 4
maxD = 2

sh = spinHalf()
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

gate = np.exp(-s)*np.sqrt(c*(1-c))*contract(np.reshape(sh.op("pu"), (2, 2, 1)), np.reshape(sh.op("x"), (2, 2, 1)), 2, 2)
gate -= (1-c) * contract(np.reshape(sh.op("pu"), (2, 2, 1)), np.reshape(sh.op("pu"), (2, 2, 1)), 2, 2)
gate -= (c) * contract(np.reshape(sh.op("pu"), (2, 2, 1)), np.reshape(sh.op("pd"), (2, 2, 1)), 2, 2)


sitesList = []
for i in range(N):
    for j in range(N):
        if j % 2 == 0 and N-j-1 > 0:
            sitesList.append([[i, j], [i, j+1]])
for i in range(N):
    for j in range(N):
        if j % 2 == 1 and N-j-1 > 0:
            sitesList.append([[i, j], [i, j+1]])
for i in range(N):
    for j in range(N):
        if i % 2 == 0 and N-i-1 > 0:
            sitesList.append([[i, j], [i+1, j]])
for i in range(N):
    for j in range(N):
        if i % 2 == 1 and N-i-1 > 0:
            sitesList.append([[i, j], [i+1, j]])
            
opList = opList2d(sh, N)
opList.add("id", [0, 0], 0)
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
    
        
lastEnergy = 0
energy = 1
for dt in [1.0, 0.1]:
    lastEnergy = 0
    gates = exp(gate, [1, 3], dt)
    while np.abs((energy - lastEnergy) / energy) > 10**-4:
        for k in range(100):
            for site in sitesList:
                psi.applyGate(gates, site, mindim=maxD, maxdim=maxD, normalize=True)
            #print("dt="+str(dt)+" sim="+str(k+1))
        print("compelte")
        #print(dotPEPS(psi, psi))
        
        env = environment(psi)
        lastEnergy = energy
        energies = np.real(expectationOpListEnv(env, opList))
        energy = np.sum(energies[1:]) / energies[0]
        
        print("dt="+str(dt)+" energy="+str(np.real(energy)))


#%%
energy = 0
for site in sitesList:
    psi2 = copy.deepcopy(psi)
    psi2.applyGate(gate, site, cutoff=10**-12)
    energy += dotPEPS(psi, psi2)

#%%
occs = np.zeros((N, N))
n = sh.op("n")
for i in range(N):
    for j in range(N):
        psi2 = copy.deepcopy(psi)
        psi2.tensors[i][j] = contract(psi2.tensors[i][j], n, 4, 1)
        occs[i, j] = np.real(dotPEPS(psi, psi2))
        

