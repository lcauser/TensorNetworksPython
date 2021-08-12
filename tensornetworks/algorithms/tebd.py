import numpy as np
import copy
from tensornetworks.tensors import *
from tensornetworks.structures.mps import *
from tensornetworks.structures.opList import opList
from tensornetworks.structures.gateList import gateList, trotterize, applyGates
from tensornetworks.algorithms.variationalMPS import variationalSum

def tebd(psi : mps, H : opList, tmax, Vs=[], dynamics='real', dt = 0.01,
         order = 2, norm=0, updates='fast', cutoff=10**-12, mindim=1,
         maxdim=0):
    
    # Determine the truncation criterion
    fullErr = cutoff
    fullDim = maxdim
    if updates == 'fast':
        gatesErr = cutoff
        gatesMax = maxdim
    elif updates == 'full':
        gatesErr = 0
        gatesMax = 0
    else:
        raise ValueError("Only Fast (svd) and full (variational) updates " \
                         + "are  supported.")
            
    # Check projector MPSs
    if not isinstance(Vs, list):
        Vs = [Vs]
    for V in Vs:
        if not isinstance(Vs, list):
            raise ValueError("Projected vectors must be an MPS.")
        else:
            if V.dim != psi.dim or V.length != psi.length:
                raise ValueError("Projected vectors must be on the same" + \
                                 " basis as psi.")
    
    # Trotterize the operator list
    gates = trotterize(H, dt, order=order)
    
    # Find the number of steps
    nsteps = int(tmax/dt)
    
    # Repeatedly evolve in time
    converged = False
    step = 0
    while not converged:
        psi = applyGates(gates, psi, mindim=mindim, maxdim=gatesMax,
                         cutoff=gatesErr)
        
        # Renormalize
        psiNorm = np.log(psi.norm())
        norm += psiNorm
        psi.normalize()
        
        # Add variational minimization for full updates and/or projection.
        if updates == 'full' or len(Vs) > 0:
            vecs = [psi]
            for V in Vs:
                vecs.append(-dot(psi, V)*V)
            psi = variationalSum(psi, vecs, mindim=mindim, maxdim=fullDim,
                                 cutoff=fullErr)
        
        # Check convergence
        step += 1
        if step >= nsteps:
            converged = True
        
        
        # Output relavent information
        print("time=" + str(round(step*dt, 10)) + \
              ", maxbonddim=" + str(psi.maxBondDim())+", norm=" + \
              str(round(np.real(psiNorm), 10)))
    
    return psi