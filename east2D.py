from tensornetworks.structures.peps import *
from tensornetworks.lattices.mps import spinHalf
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.lattices.mps import spinHalf
import copy

# Parameters
c = 0.5
s = 0
N = 4
maxD = 3

sh = spinHalf()
states = []
for i in range(N):
    states2 = []
    for j in range(N):
        states2.append("s")
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
        


for dt in [10, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    gates = exp(gate, [1, 3], dt)
    for k in range(min(500, int(5000 / dt))):
        for site in sitesList:
            psi.applyGate(gates, site, maxdim=maxD, normalize=True)
        print("dt="+str(dt)+" sim="+str(k+1))
    psi.normalize()
    #print(dotPEPS(psi, psi))

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
        

