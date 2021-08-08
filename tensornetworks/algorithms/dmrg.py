import numpy as np
import copy
from tensornetworks.tensors import *
from tensornetworks.structures.mps import *
from tensornetworks.structures.mpo import *
from tensornetworks.structures.projMPSMPO import *
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator # Creates a linear op 

def dmrg(H : mpo, psi0 : mps, Vs=[], eps=10**-12, maxsweeps=100, minsweeps=1):
    psi = copy.deepcopy(psi0).orthogonalize(0)
    
    # Find the projected MPO-MPS products
    projH = projMPSMPO(H, psi)
    
    rev = 0
    for i in range(100):
        for j in range(psi0.length-1):
            # Determine the site
            site = j if not rev else psi0.length - 1 - j           
            
            # Get the tensor at the sites
            A1 = psi.tensors[site if not rev else site - 1]
            A2 = psi.tensors[site + 1 if not rev else site]
            A0 = contract(A1, A2, 2, 0)
            
            # Fetch the projected matrices
            Heff = LinearOperator(projH.opShape(), matvec= lambda x: projH._matvec(x))
            eig, vec = eigsh(Heff, which='SA', k = 1, v0=A0.flatten(), ncv=3, tol=0.01)


            A = np.reshape(vec, np.shape(A0))
            psi.replacebond(site - rev, A, rev, cutoff=eps)   
            
            # Build projector blocks
            if not rev:
                projH.buildLeft(site)
                projH.center = site + 1
            else:
                projH.buildRight(site)
                projH.center = site - 1
            
        rev = 1 - rev
        projH.rev = rev
        
        print("Sweep "+str(i+1)+", energy="+str(eig), "maxbonddim="+str(psi.maxBondDim()))
    
    
    return psi