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

def vbMPO(O : mpo, chi, tol=10**-5, cutoff=0, maxiter=1000):
    """
    Variationally minimize a boundary MPO for some given environment bond
    dimension.

    Parameters
    ----------
    O : mpo
    chi : int
    tol : float, optional
        Change in cost per sweep before converging. The default is 10**-8.
    cutoff : Float, optional
        Truncation error. The default is 0.
    maxiter : int, optional
        The maximum number of sweeps.. The default is 1000.

    Returns
    -------
    P : mpo
        Truncated boundary mpo.

    """
    
    # Start by copying the bMPO and using SVD to truncate
    P = copy.deepcopy(O)
    P.orthogonalize(P.length-1)
    norm = P.norm()**2
    P.orthogonalize(0, maxdim=chi, cutoff=cutoff)
    
    # Construct the projectors
    projOP = projbMPO(O, P)
    
    # Calculate the difference (up to a constant..)
    projOPdiff = projOP.calculate()
    cost = norm + P.norm()**2 - projOPdiff - np.conj(projOPdiff)
    
    
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
            projOP.moveCenter(site)
            
            # Find the optimal update
            A = projOP.vector()
            A = 0.5 * (A + dag(A))
            P.tensors[site] = A
        
        # Reverse direction
        rev = 1 - rev
        
        # Calculate the new difference
        oldCost = cost
        projOPdiff = projOP.calculate()
        cost = norm + P.norm()**2 - projOPdiff - np.conj(projOPdiff)
        
        # Check to see if converged
        iterations += 1
        if np.abs(np.real(cost)) >= 10**-12:
            if np.real((oldCost - cost)/cost) <= tol:
                converged = True
        else:
            converged = True
        if maxiter != 0 and iterations >= maxiter:
            converged = True
    return P