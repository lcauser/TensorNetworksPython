import numpy as np
from structures.tensors import *
import copy

c = 0.5
s = -0.1
cutoff = 0
maxDim = 400
timestep = 0.01
time = 1

# Create starting tensors
psi0 = np.zeros((1, 2, 1))
psi0[0, 0, 0] = np.sqrt(c)
psi0[0, 1, 0] = np.sqrt(1-c)
psi0 = tensor((1, 2, 1), psi0)
S = tensor((1, 1), np.array([[1]]))
A = copy.deepcopy(psi0)
Al = copy.deepcopy(S)
B = copy.deepcopy(psi0)
Bl = copy.deepcopy(S)

# Create unitary gate
n = np.zeros((2, 2))
n[0, 0] = 1
x = np.zeros((2, 2))
x[0, 1] = 1
x[1, 0] = 1
gate = np.exp(-s)*np.sqrt(c*(1-c))*x - c*np.eye(2) - (1-2*c)*n
storage1 = np.zeros((2, 2, 2))
storage1[:, :, 0] = n
storage1[:, :, 1] = gate
storage2 = np.zeros((2, 2, 2))
storage2[0, :, :] = gate
storage2[1, :, :] = n
ten1 = tensor((2, 2, 2), storage1)
ten2 = tensor((2, 2, 2), storage2)
ten = contract(ten1, ten2, 2, 0)
tenexp = exp(ten, [1, 3], timestep)

steps = int(time / timestep)
normal = 0
partitions = []
for i in range(steps):
    for j in range(2):
        if j == 0:
            ten1 = A
            ten2 = B
            ten1l = Al
            ten2l = Bl
        else:
            ten1 = B
            ten2 = A
            ten1l = Bl
            ten2l = Al
        
        # Apply gate
        ten = contract(ten2l, ten1, 1, 0)
        ten = contract(ten, ten1l, 2, 0)
        ten = contract(ten, ten2, 2, 0)
        ten = contract(ten, ten2l, 3, 0)
        ten = contract(ten, tenexp, 1, 0)
        ten = trace(ten, 1, 4)
        ten = permute(ten, 1, 3)
        
        # SVD to decompose
        dims = ten.dims
        ten, C1 = combineIdxs(ten, [0, 1])
        ten, C2 = combineIdxs(ten, [0, 1])
        U, S, V = svd(ten, 1, 1, maxDim, cutoff)
        U = reshape(U, (dims[0], dims[1], S.dims[0]))
        V = reshape(V, (S.dims[1], dims[2], dims[3]))
        
        # Pull out end singular values
        mat = np.diag(ten2l.storage)
        invMat = tensor(ten2l.dims, np.diag(1 / mat))
        U = contract(invMat, U, 1, 0)
        V = contract(V, invMat, 2, 0)
        
        # Renormalise
        #norm1 = norm(U)
        #U.storage = U.storage / norm1
        #norm2 = norm(V)
        #V.storage = V.storage / norm2
        #norm3 = norm(S)
        #S.storage = S.storage / norm3
        #normal += np.log(norm1) + np.log(norm2) + np.log(norm3)
        #print("----------")
        #print(np.log(norm1))
        #print(np.log(norm2))
        #print(np.log(norm3))
    
        
        # Update MPS
        if j == 0:
            Al = S
            A = U
            B = V
        else:
            Bl = S
            A = V
            B = U
        
    # Store partition sum
    ten = trace(contract(A, A, 0, 0), 0, 2)
    ten = contract(contract(ten, Al, 0, 0), Al, 0, 0)
    ten = contract(contract(ten, B, 0, 0), B, 0, 0)
    ten = contract(contract(ten, Bl, 1, 0), Bl, 2, 0)
    ten = trace(trace(ten, 2, 3), 0, 1)
    print(np.log(ten.storage) + 2*normal)
    #print(normal)
        
        
                
        