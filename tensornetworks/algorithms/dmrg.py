import numpy as np
import copy
from tensornetworks.tensors import *
from tensornetworks.structures.mps import *
from tensornetworks.structures.mpo import *
from tensornetworks.structures.projMPSMPO import *
from tensornetworks.structures.projMPS import *
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator # Creates a linear op 

def dmrg(Hs, psi0 : mps, Vs=[], maxsweeps=0, minsweeps=1, tol=10**-8,
         numconverges=4, cutoff=10**-12, maxdim=0, mindim=1):
    
    # Check to see if there is some truncation parameters
    if cutoff == 0 and maxdim == 0:
        print("Warning: there is no truncation criteria and bond dimensions" \
              +" grow with no restrictions.")
    
    # Check to see if there is a convergence criteria
    if maxsweeps == 0 and tol == 0:
        print("Warning: There is no convergence criteria to limit the number" \
              + " of sweeps.")
    
    # Handle the initial MPS
    if not isinstance(psi0, mps):
        print("Error: psi0 must be an MPS.")
        return None
    psi = copy.deepcopy(psi0).orthogonalize(0)
    maxbonddim = psi.maxBondDim()
    dim = psi.dim
    
    # Check the list of MPOs
    projHs = []
    if not isinstance(Hs, list):
        Hs = [Hs]
    for H in Hs:
        if not isinstance(H, mpo):
            print("Error: Hs must be an MPO or list of MPOs.")
            return None
        if H.dim != dim:
            print("Error: All tensor objects must have the same local dimensions")
            return None
        projHs.append(projMPSMPO(H, psi))
        
    # Check the list of projected MPSs
    if not isinstance(Vs, list):
        Vs = [Vs]
    for V in Vs:
        if not isinstance(V, mps):
            print("Error: Hs must be an MPS or list of MPSs.")
            return None
        if V.dim != dim:
            print("Error: All tensor objects must have the same local dimensions")
            return None
        projHs.append(projMPS(V, psi, squared=True))
    
    
    rev = 0 # Sweeps reversed?
    converge = False # Check convergence
    convergedSweeps = 0 # Number of sweeps the convergence criteria is satisfied
    sweeps = 0 # Number of sweeps
    energy = 0 # Measured energy
    
    while not converge:
        for j in range(psi0.length-1):
            # Determine the site
            site = j if not rev else psi0.length - 1 - j           
            
            # Get the tensor at the sites
            A1 = psi.tensors[site if not rev else site - 1]
            A2 = psi.tensors[site + 1 if not rev else site]
            A0 = contract(A1, A2, 2, 0)
            
            # Calculate the effective Hamiltonian
            Heffs = []
            mv = lambda x: sum(projHs[i]._matvec(x) for i in range(len(projHs)))
            Heff = LinearOperator(projHs[0].opShape(),  matvec = mv)
            #print(A0)
            # Optimize the bond
            eig, vec = eigsh(Heff, which='SA', k = 1, v0=A0.flatten(), ncv=3,
                             tol=0.1)
            A = np.reshape(vec, np.shape(A0))
            psi.replacebond(site - rev, A, rev, cutoff=cutoff, mindim=mindim,
                            maxdim=maxdim)   
            #print(eig)
            #print(vec)
            
            # Build projector blocks
            for projH in projHs:
                if not rev:
                    projH.buildLeft(site)
                    projH.center = site + 1
                else:
                    projH.buildRight(site)
                    projH.center = site - 1
                    
        # Reverse the sweep direction
        rev = 1 - rev
        for projH in projHs:
            projH.rev = rev
        
        # Output sweep information
        sweeps += 1
        print("Sweep "+str(sweeps)+", energy="+str(eig), "maxbonddim="+str(psi.maxBondDim()))
        
        # Check convergence
        if sweeps >= minsweeps:
            if sweeps >= maxsweeps and maxsweeps != 0:
                converge = True
            
            if np.abs(energy - eig) < tol and maxbonddim == psi.maxBondDim():
                convergedSweeps += 1
            else:
                convergedSweeps = 0
                
            if convergedSweeps >= numconverges:
                converge = True
        
        # Update information
        energy = eig
        maxbonddim = psi.maxBondDim()
        
    
    return psi