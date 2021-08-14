"""
    Time-evolving block decimation repeadibly evolves a MPS by applying
    sequential trotter gates, using SVD to split the tensor back into the sites
    and truncate. This module implements this, with additional options to 
    apply full updates through variational minimzation, and the ability to 
    add custom observers to measure properties of the MPS throughout the 
    evolution.
"""

import numpy as np
import copy
from tensornetworks.tensors import *
from tensornetworks.structures.mps import *
from tensornetworks.structures.opList import opList, opExpectation
from tensornetworks.structures.gateList import gateList, trotterize, applyGates
from tensornetworks.algorithms.variationalMPS import variationalSum


def tebd(psi : mps, H : opList, tmax, Vs=[], observers=[], dynamics='real', 
         dt = 0.01, save = 0.1, order = 2, norm=0, updates='fast',
         mindim=1, maxdim=0, cutoff=10**-12):
    """
    Evolve an MPS in time exponetially according to some operator.

    Parameters
    ----------
    psi : mps
        Initial state
    H : opList
        List of terms in the operator (e.g. hamiltonian)
    tmax : float
        The total time to evolve for.
    Vs : list, optional
        List of MPS to project out of psi. The default is [].
    observers: list, optional
        List of observer classes. The default is [].
    dynamics : str, optional
        'real' (imaginary) or 'quantum'. The default is 'real'.
    dt : float, optional
        The timestep in evolution gates. The default is 0.01.
    save : float, optional
        The time which to measure observables. The default is 0.1.
    order : int, optional
        Trotter order, 1 or 2. The default is 2.
    norm : int, optional
        Some initial log norm for psi. The default is 0.
    updates : str, optional
        'fast' (SVD) or 'full' (variational). The default is 'fast'.
    mindim : int, optional
        The minimum number of singular values to keep. The default is 1.
    maxdim : int, optional
        The maximum number of singular values to keep. The default is 0,
        which defines no upper limit.
    cutoff : float, optional
        The truncation error of singular values. The default is 0.

    Returns
    -------
    psi : mps
        Evolved MPS.

    """
    
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
    
    # Check observers are valid and peform initial measurements
    if not isinstance(observers, list):
        observers = [observers]
    for observer in observers:
        if not isinstance(observer, TEBDobserver):
            raise ValueError("Observer must be a child of the TEBDobserver class.")
        observer.observe(0, psi, norm)
    
    # Trotterize the operator list
    if dynamics == 'quantum':
        H = -1j*H
    gates = trotterize(H, dt, order=order)
    
    # Check the savetime
    if save < dt:
        print("Warning: save time is more than evolution time. Setting it " + \
              "to be the same.")
        save = dt
    
    # Find the number of steps
    nsteps = int(tmax/dt)
    nsave = int(save/dt)
    
    # Repeatedly evolve in time
    converged = False
    step = 0
    while not converged:
        # Apply gates
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
        
        # Increase steps and measure observables
        step += 1
        for observer in observers:
            observer.observe(step*dt, psi, norm)
        
        # Check convergence      
        if step >= nsteps:
            converged = True
        for observer in observers:
            if observer.checkdone():
                converged = True
              
        # Output relavent information
        print("time=" + str(round(step*dt, 10)) + \
              ", maxbonddim=" + str(psi.maxBondDim())+", norm=" + \
              str(round(np.real(psiNorm), 10)))
    
    return psi


class TEBDobserver:
    
    def __init__(self):
        """ An observer class can be plugged into TEBD to take measurements,
        and check for convergence. """
        self.times = []
        self.measurements = []
    
    def observe(self, time, psi, norm):
        """
        Take a measurement.

        Parameters
        ----------
        time : float
        psi : mps
        norm : float

        """
        self.times.append(time)
        self.measurements.append(self.measure(psi, norm))
        return None
    
    def measure(self, psi, norm):
        """
        Measure something,

        Parameters
        ----------
        psi : mps
        norm : float

        Returns
        -------
        measurement

        """
        return None
    
    def checkdone(self):
        """ Check for convergence. """
        return False


class normTEBDobserver(TEBDobserver):
    
    def __init__(self, tol=0):
        """
        Measures the normal, and can stop early if the difference in norm
        is unchanging.

        Parameters
        ----------
        tol : float
            The tolerance for change in change of norm.

        """
        super().__init__()
        self.tol = tol
    
    
    def measure(self, psi, norm):
        return norm
    
    def checkdone(self):
        if len(self.measurements) > 2 and self.tol > 0:
            # Measure differences in norm
            diff1 = np.abs(self.measurements[-1] - self.measurements[-2])
            diff2 = np.abs(self.measurements[-2] - self.measurements[-3])
            
            # Measure their difference
            diff = np.abs(diff2 - diff1)
            
            # Check the condition
            if diff < self.tol:
                return True
        return False
    
    
class oplistTEBDobserver(TEBDobserver):
    
    def __init__(self, ops : opList):
        """
        Measures a list of operators.

        Parameters
        ----------
        ops : opList
            List of operators

        """
        super().__init__()
        self.ops = ops
    
    def measure(self, psi, norm):
        return opExpectation(self.ops, psi)
    