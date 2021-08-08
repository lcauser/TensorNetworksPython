# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
#from stuctures.tensors import *
#from structures.mps import *
#from structures.projMPS import *
from scipy.sparse.linalg import LinearOperator, spsolve
from scipy.sparse.linalg import cg

def variationalSum(psi0, Vs, minsweeps=4, maxsweeps=1000, mindim=1, maxdim=0, cutoff=0):
    # Check to see if Vs is list
    if not isinstance(Vs, list):
        Vs = [Vs]
    
    # Make psi orthogonal at the first site
    psi = copy.deepcopy(psi0).truncate(mindim, maxdim, cutoff)
    psi.orthogonalize(0)
    
    # Get the projectors for each term and projector
    projVs = [projMPS(psi, V) for V in Vs]
    
    # Loop through until converged
    converged = False
    site = 0
    sweeps = 0
    rev = False
    prevCost = 0
    lastBondDim = psi.maxBondDim()
    while not converged:
        for hs in range(2):
            for bond in range(psi.length-1):
                # Define the effective operator at the sites
                site1 = site if not rev else site - 1
                site2 = site+1 if not rev else site
                dims = (psi.tensors[site1].dims[0], psi.dim, psi.dim, psi.tensors[site2].dims[2])
                dim = np.prod(dims)
                
                # Find the effective vector
                i = 0 
                for projV in projVs:
                    if i == 0:
                        vec = projV._matvec(0)
                    else:
                        vec += projV._matvec(0)
                    i += 1
                
                
                # Find the initial vector
                A = contract(psi.tensors[site1], psi.tensors[site2], 2, 0)
                A = A.storage.flatten()
                
                # Find the optimal solution
                A = vec
                
                # Replace the bond
                A = tensor(dims, np.reshape(A, dims))
                psi.replacebond(site1, A, rev, mindim, maxdim, cutoff)
                cost = norm(A)
                
                # Move to next site
                site = site + 1 if not rev else site - 1
                for projV in projVs:
                    projV.moveCenter(site)
            
            # Reverse direction
            site = psi.length - 1 if not rev else 0
            rev = not rev
            for projV in projVs:
                projV.moveCenter(site)
                projV.rev = rev
        
        # Increase Sweeps
        sweeps += 1
        if sweeps >= maxsweeps:
            converged = True
        
        diff = np.abs((prevCost - cost) / cost)
        prevCost = cost
        bondDim = psi.maxBondDim()
        # Calculate overlap
        if sweeps >= minsweeps:
            if diff < 10**-12:
                if lastBondDim == bondDim:
                    converged = True
        lastBondDim = bondDim
        
        #print("Sweep "+str(sweeps)+" cost="+str(diff)+" maxbonddim="+str(bondDim))
        
    return psi
                