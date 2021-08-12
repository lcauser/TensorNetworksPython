from tensornetworks.structures.mps import *
from tensornetworks.structures.mpo import *
from tensornetworks.lattices.mps import spinHalf
import numpy as np
import tensornetworks.tensors as tn
from tensornetworks.structures.opList import opList, opExpectation, applyOp
from tensornetworks.structures.gateList import gateList, trotterize, applyGates
from tensornetworks.algorithms.qjmc import qjmcSimulation, qjmcGates, qjmc
import matplotlib.pyplot as plt

N = 40
kappa = 1
gamma = 0
omega = 1
dt = 0.01
tmax = 100.0
saveTime = 0.1

# Find the "light" and "dark states"
ss = (8*omega**2+(kappa+gamma)**2)**-1 * \
      np.array([[4*omega**2 + gamma*(kappa+gamma), -2j*omega*(kappa-gamma)],
                [2j*omega*(kappa-gamma), 4*omega**2 + kappa*(kappa+gamma)]])
eigs, vecs = np.linalg.eig(ss)
lightEig = eigs[np.argmin(eigs)]
lightVec = vecs[:, np.argmin(eigs)]
darkEig = eigs[np.argmax(eigs)]
darkVec = vecs[:, np.argmax(eigs)]

# Create the vector sspace
sh = spinHalf()
sh.addState("light", tensor(2, lightVec))
sh.addState("dark", tensor(2, darkVec))
sh.addOp("plight", tensor((2, 2), np.outer(lightVec, np.conj(lightVec))), "plight")
sh.addOp("pdark", tensor((2, 2), np.outer(darkVec, np.conj(darkVec))), "pdark")

# Construct some initial state
states = []
for i in range(N):
    r = np.random.rand()
    if r < np.real(darkEig):
        states.append("dark")
    else:
        states.append("light")
#states = ["dark"] * N
psi0 = productMPS(sh, states)
psi0.orthogonalize(0)

# Create the Hamiltonian
H = opList(sh, N)
H.add(["x"], [0], omega)
for i in range(N-1):
    H.add(["plight", "x"], [i, i+1], omega)
    H.add(["x", "plight"], [i, i+1], omega)
#H.add(["x"], [N-1], omega)

# Create the jump operators list
jumpOps = opList(sh, N)
jumpOps.add(["s-"], [0], np.sqrt(kappa))
jumpOps.add(["s+"], [0], np.sqrt(gamma))
for i in range(N-1):
    jumpOps.add(["plight", "s-"], [i, i+1], np.sqrt(kappa))
    jumpOps.add(["plight", "s+"], [i, i+1], np.sqrt(gamma))
    jumpOps.add(["s-", "plight"], [i, i+1], np.sqrt(kappa))
    jumpOps.add(["s+", "plight"], [i, i+1], np.sqrt(gamma))
#jumpOps.add(["s-"], [N-1], np.sqrt(kappa))
#jumpOps.add(["s+"], [N-1], np.sqrt(gamma))


# Create code for observers
observers = opList(sh, N)
for i in range(N):
    observers.add("pu", i)
for i in range(N):
    observers.add("plight", i)

acts, observations, times = qjmc(H, jumpOps, psi0, dt, tmax, 1, saveTime, observers)

plt.pcolormesh(times, np.linspace(1, N, N+1), np.transpose(observations[:, N:]))
plt.show()
plt.pcolormesh(times, np.linspace(1, N, N+1), np.transpose(observations[:, 0:N]), cmap='gray_r')
