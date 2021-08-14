from tensornetworks.structures.peps import *
from tensornetworks.lattices.mps import spinHalf
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.lattices.mps import spinHalf
import copy

# Parameters
c = 0.5
s = 1.0
N = 4
maxD = 2
chi = 16

dt = 0.1
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

gates = exp(gate, [1, 3], dt)

#%% Create the bottom environment blocks
blocks = [0] * N
for i in range(N-1):
    block = []
    for j in range(N):
        # Get the old block tensor
        if i == 0:
            B = np.ones((1, 1, 1, 1))
        else:
            B = blocks[N-i][j]
        
        # Get the next joined tensor
        A = psi.tensors[N-1-i][j]
        
        # Contract
        B = contract(B, dag(A), 1, 2)
        B = contract(B, A, 1, 2)
        B = trace(B, 5, 9)
        B, cmb = combineIdxs(B, [0, 2, 5])
        B, cmb = combineIdxs(B, [0, 2, 4])
        B = permute(B, 2, 0)
        
        # Add to 'MPS'        
        block.append(B)
    blocks[N-1-i] = block