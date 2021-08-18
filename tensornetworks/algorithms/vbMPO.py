"""
    Perform variational minimization on a boundary MPO to find a bMPO with a
    smaller bond dimension. Useful in building environments for PEPS.
    
    Future updates:
        It might be worth doing two-site optimization with a dynamic
        bond dimension, as some results show that smaller bond dimensions give
        the same result.
"""

# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.mpo import mpo
from tensornetworks.structures.projbMPO import projbMPO

def vbMPO(O : mpo, chi, tol=10**-8, cutoff=10**-20, maxiter=100):
    # Start by copying the bMPO and using SVD to truncate
    P = copy.deepcopy(O)
    P.orthogonalize(P.length-1)
    P.orthogonalize(0, maxdim=chi, cutoff=cutoff)
    
    # Construct the projectors
    projPP = projbMPO(P, P)
    projOP = projbMPO(O, P)
    
    # Calculate the difference (up to a constant..)
    projOPdiff = projOP.calculate()
    diff = 2*projPP.calculate() - projOPdiff - np.conj(projOPdiff)
    
    # Loop until the change in difference converges
    converged = False
    rev = 0
    iterations = 0
    while not converged:
        # Loop through each site
        for i in range(O.length):
            # Determine the site
            site = i if not rev else O.length - 1 - i
            
            # Orthogonalize bMPO and move projection centers
            P.orthogonalize(site)
            projPP.moveCenter(site)
            projOP.moveCenter(site)
            
            # Find the optimal update
            A = projOP.vector()
            A = 0.5 * (A + dag(A))
            P.tensors[site] = A
        
        # Reverse direction
        rev = 1 - rev
        
        # Calculate the new difference
        oldDiff = diff
        projOPdiff = projOP.calculate()
        diff = 2*projPP.calculate() - projOPdiff - np.conj(projOPdiff)
        
        # Check to see if converged
        iterations += 1
        if np.abs(diff - oldDiff) < tol:
            converged = True
        if maxiter != 0 and iterations >= maxiter:
            converged = True
    
    return P