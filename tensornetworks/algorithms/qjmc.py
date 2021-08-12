from tensornetworks.structures.opList import opList, opExpectation, applyOp
from tensornetworks.structures.gateList import gateList, trotterize, applyGates
from tensornetworks.tensors import *
from tensornetworks.structures.mps import mps
import numpy as np
import copy

def qjmc(H : opList, jumpOps : opList, psi0 : mps, dt, tmax, numsims, save = 0,
         observers = 0, cutoff=10**-12, mindim=1, maxdim=0, verbose=1):
    actCount = 0
    if not observers is None:
        times = np.linspace(0, tmax, int(tmax / save) + 1)
        observations = np.zeros((np.size(times), len(observers.ops)))
        
    escapeOps, gates = qjmcGates(H, jumpOps, dt)
    
    for sim in range(numsims):
        data = qjmcSimulation(gates, jumpOps, escapeOps, copy.deepcopy(psi0),
                              tmax, dt, save, observers, cutoff, mindim, maxdim)
        actCount += len(data[1])
        if not observers is None:
            observations += np.real(np.asarray(data[3]))
            
        print("Simulation "+str(sim+1)+" completed.")
        
    if not observers is None:
        return actCount / numsims, observations / numsims, times
    return actCount / numsims


def qjmcGates(H : opList, jumpOps : opList, dt):
    # Create the escape operators from the jump operators
    escapeOps = opList(jumpOps.sitetype, jumpOps.length)
    for i in range(len(jumpOps.ops)):
        sites = jumpOps.sites[i]
        coeff = jumpOps.coeffs[i] * dag(jumpOps.coeffs[i])
        ops = []
        for op in jumpOps.ops[i]:
            ops.append(jumpOps.sitetype.opProd([jumpOps.sitetype.dagger(op), op]))
        escapeOps.add(ops, sites, coeff)
    
    # Create the effectively Hamiltonian and the bond gates
    Heff = -1j*H+ (-0.5*escapeOps)
    gates = trotterize(Heff, dt, 2)
    
    return escapeOps, gates


def qjmcSimulation(gates : gateList, jumpOps : opList, escapeOps : opList,
                   psi : mps, tmax, dt, save=0, observers=None,
                   cutoff=10**-12, mindim=1, maxdim=0, verbose=1):
    # If save is zero, change it to the timestep
    if save == 0:
        save = dt
    
    # Determine the number of steps
    steps = int(tmax / dt)
    saveSteps = int(save / dt)
    
    # Store quantum jumps
    jumps = []
    jumpTimes = []
    
    # Initiate time and initial observations
    time = 0
    if not observers is None:
        observations = [opExpectation(observers, psi)]
    
    # Loop through each step
    for i in range(steps):
        # Apply the evolution gates and increase time
        psi = applyGates(gates, psi, cutoff=cutoff, maxdim=maxdim,
                         mindim=mindim)
        time += dt
        
        # Measure the norm and determine if a jump occurs
        norm = psi.norm()**2
        r = np.random.rand()
        if r > norm:
            # Calculate jump rates and pick a jump
            rates = opExpectation(escapeOps, psi)
            rates = np.real(rates / np.sum(rates))
            idx = np.where(np.random.rand() < np.cumsum(rates))[0][0]
            
            # Do the quantum jump
            psi = applyOp(psi, jumpOps.sitetype, jumpOps.ops[idx],
                          jumpOps.sites[idx], jumpOps.coeffs[idx])
            psi.orthogonalize(0)
            
            # Store the jump information
            jumps.append(idx)
            jumpTimes.append(time)
            
        # Re normalize
        psi.normalize()
        
        # Measure observations
        if not observers is None:
            if (i+1) % saveSteps == 0:
                observations.append(opExpectation(observers, psi))
        
        # Output information
        if verbose:
            print("time="+str(round(time, 5)) + ", maxlinkdim=" + \
                  str(psi.maxBondDim()) + " jumps=" + str(len(jumps)))
        
    if not observers is None:
        return psi, jumps, jumpTimes, observations
    
    return psi, jumps, jumpTimes