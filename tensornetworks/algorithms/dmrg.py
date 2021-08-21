"""
    Matrix product states provide a variational class underlying the 
    density matrix renormalization group (DMRG). This algorithm performs the
    DMRG on an MPS given a list of MPOs, and projects away from MPSs.
"""

import numpy as np
import copy
from tensornetworks.tensors import *
from tensornetworks.structures.mps import mps
from tensornetworks.structures.mpo import mpo
from tensornetworks.structures.projMPSMPO import projMPSMPO
from tensornetworks.structures.projMPS import projMPS
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse.linalg import LinearOperator # Creates a linear op 
from scipy.linalg import eig as speig

def dmrg(Hs, psi0 : mps, Vs=[], maxsweeps=0, minsweeps=1, tol=10**-8,
         numconverges=4, cutoff=10**-12, maxdim=0, mindim=1, hermitian=True,
         nsites=2):
    """
    Peform the DMRG for a list of MPOs, and projection MPSs.

    Parameters
    ----------
    Hs : List
        List of MPOs.
    psi0 : mps
        Initial MPS to initiate DMRG.
    Vs : list, optional
        List of projection MPSs. The default is [].
    maxsweeps : int, optional
        The maximum number of sweeps. The default is 0.
    minsweeps : int, optional
        The minimum number of sweeps. The default is 1.
    tol : float, optional
        The tolerence for change in energy per sweep. The default is 10**-8.
    numconverges : int, optional
        The number of sweeps for which the convergence criteria must be 
        satisfied. The default is 4.
    cutoff : float, optional
        The tolerated truncation error. The default is 10**-12.
    maxdim : int, optional
        The maximum bond dimension. The default is 0 (unrestricted).
    mindim : int, optional
        The minimum bond dimension. The default is 1.
    hermitian : bool, optional
        Is the problem hermitian? The default is true.
    nsites : int, optional
        The number of sites to optimize. The default is 2.

    Returns
    -------
    psi : mps
        Optimized MPS.

    """
    
    # Check to see if there is some truncation parameters
    if cutoff == 0 and maxdim == 0:
        print("Warning: there is no truncation criteria and bond dimensions" \
              + " grow with no restrictions.")
    
    # Check to see if there is a convergence criteria
    if maxsweeps == 0 and tol == 0:
        print("Warning: There is no convergence criteria to limit the number" \
              + " of sweeps.")
    
    # Check the number of sites to give warning.
    if nsites == 1:
        print("Warning: nsites=1 corresponds to single-site DMRG, which " + \
              "does not have a dynamical bond dimension.")
    
    # Handle the initial MPS
    if not isinstance(psi0, mps):
        raise ValueError("Error: psi0 must be an MPS.")
    psi = copy.deepcopy(psi0).orthogonalize(0)
    maxbonddim = psi.maxBondDim()
    dim = psi.dim
    
    # Check the list of MPOs
    projHs = []
    if not isinstance(Hs, list):
        Hs = [Hs]
    for H in Hs:
        if not isinstance(H, mpo):
            raise ValueError("Error: Hs must be an MPO or list of MPOs.")
        if H.dim != dim:
            raise ValueError("Error: All tensor objects must have the same" \
                             + " local dimensions")
        projHs.append(projMPSMPO(H, psi, nsites=nsites))
        
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
        projHs.append(projMPS(V, psi, squared=True, nsites=nsites))
    
    
    rev = 0 # Sweeps reversed?
    converge = False # Check convergence
    convergedSweeps = 0 # Number of sweeps the convergence criteria is satisfied
    sweeps = 0 # Number of sweeps
    energy = 0 # Measured energy
    
    while not converge:
        for j in range(psi0.length+1-nsites):
            # Determine the site
            site = j if not rev else psi0.length - 1 - j           
            
            # Get the tensor at the sites and contract
            site1 = site if not rev else site - nsites + 1
            A0 = psi.tensors[site1]
            for i in range(nsites - 1):
                A1 = psi.tensors[site1+1+i]
                A0 = contract(A0, A1, 2+i, 0)
            
            # Calculate the effective Hamiltonian
            mv = lambda x: sum(projHs[i]._matvec(x) for i in range(len(projHs)))
            Heff = LinearOperator(projHs[0].opShape(),  matvec = mv)
            
            # Deal with single-site struggles
            if Heff.shape[0] == 2:
                # Deal with case scipy apparently cannot
                mv = lambda x: sum(np.real(projHs[i]._matvec(x)) for i in range(len(projHs)))
                Heff = LinearOperator(projHs[0].opShape(),  matvec = mv)
                
            # Optimize the bond
            #print("--------")
            #print(np.size(A0))
            #print(Heff.shape)
            if hermitian:
                eig, vec = eigsh(Heff, k=1, which='SA', v0=A0.flatten(),
                                 ncv=3, tol=0.1)
            else:
                eig, vec = eigs(Heff, k=1, which='SR', v0=A0.flatten(),
                                ncv=3, tol=0.1)
            A = reshape(vec, shape(A0))
            psi.replacebond(site - rev*(nsites-1), A, rev, cutoff=cutoff, mindim=mindim,
                            maxdim=maxdim, nsites=nsites)   
                
            # Build projector blocks
            for projH in projHs:
                if not rev:
                    projH.buildLeft(site)
                    projH.center = min(psi.length-1, site + 1)
                else:
                    projH.buildRight(site)
                    projH.center = max(0, site - 1)
                    
        # Reverse the sweep direction
        rev = 1 - rev
        for projH in projHs:
            projH.rev = rev
            if rev == 0:
                projH.moveCenter(0)
            else:
                projH.moveCenter(psi.length-1)
            
        
        # Output sweep information
        sweeps += 1
        print("Sweep "+str(sweeps)+", energy="+str(eig) + " maxbonddim=" + \
              str(psi.maxBondDim()))
        
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