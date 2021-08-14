"""
    Gate Lists contains gates which can applied to multiple neighbouring sites
    of an MPS, in a specific order.
"""


# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.sitetypes import sitetypes
from tensornetworks.algorithms.dmrg import dmrg
from tensornetworks.structures.opList import opList
from tensornetworks.structures.mps import mps

class gateList:
    
    def __init__(self, length):
        """
        Create a gatelist to apply to an MPS.

        Parameters
        ----------
        length : int
            length of MPS.
            
        """
        # Store information
        self.length = length
        self.sites = []
        self.gates = []
        return None
    
    
    def add(self, gates, sites):
        """
        Add gates to the gate list.

        Parameters
        ----------
        gates : list
            List of tensors
        sites : list
            List of site integers

        """
        if not isinstance(gates, list):
            gates = [gates]
        if not isinstance(sites, list):
            sites = [sites]
        
        # Check the length of ops and sites are the same
        if len(gates) != len(sites):
            raise ValueError("The operator and site list must be of equal size.")
                
        for site in sites:
            if not isinstance(site, int):
                raise ValueError("The site must be an integer.")
        
        
        # Sort the operator and site list
        gates = [gate for _, gate in sorted(zip(sites, gates))]
        sites = sorted(sites)
        
        # Check to ensure there is no overlap of gates
        for i in range(len(sites)-1):
            if sites[i] + gateSize(gates[i]) - 1 >= sites[i+1]:
                raise ValueError("The bond gates in one row cannot overlap.")
        if sites[-1] + gateSize(gates[-1]) - 1 > self.length-1:
                raise ValueError("The bond gates in one row cannot overlap.")
                
        # Add the row to the list of gates
        self.sites.append(copy.deepcopy(sites))
        self.gates.append(copy.deepcopy(gates))
        
        return self
    

def trotterize(ops : opList, timestep : float, order : int = 1):
    """
    Takes an operator list and calcualtes the exponential via a trotter-suzuki
    decomposition.

    Parameters
    ----------
    ops : opList
        List of operators
    timestep : float
        Evolution timestep
    order : int, optional
        Trotter order (only 1 or 2 supported). The default is 1.

    Returns
    -------
    gl : gateList
        List of trotter gates.

    """
    # Create the gatelist and find the interaction range
    gl = gateList(ops.length)
    rng = ops.siteRange()
    
    # Decide on which order to use; if range is one, no point using more than one.
    if rng == 1:
        order = 1
        
    # Add the gates according to trotter decomposition
    if order == 1:
        for i in range(rng):
            gates = []
            sites = []
            site = i
            while site < ops.length:
                gate = ops.siteTensor(site)
                if type(gate) == np.ndarray:
                    sites.append(site)
                    gates.append(exp(timestep*gate, [2*i+1 for i in range(gateSize(gate))]))
                site += rng
            
            gl.add(gates, sites)
    elif order == 2:
        for i in range(rng):
            gates = []
            sites = []
            site = i
            ts = timestep if i == rng - 1 else timestep / 2
            while site < ops.length:
                gate = ops.siteTensor(site)
                if type(gate) == np.ndarray:
                    sites.append(site)
                    gates.append(exp(ts*gate, [2*i+1 for i in range(gateSize(gate))]))
                site += rng
            
            gl.add(gates, sites)
        
        # All backwards
        for i in range(rng - 1):
            gl.add(gl.gates[rng-2-i], gl.sites[rng-2-i])
            
    return gl


def gateSize(gate):
    """
    Calculate the range of sites acted on by a gate.

    Parameters
    ----------
    gate : np.ndarray
        Tensor gate.

    Returns
    -------
    size: int
        Number of sites.

    """
    return int(len(np.shape(gate))/2)


def applyGate(gate, site, psi : mps, rev=False, mindim=1, maxdim=0, cutoff=0):
    """
    Apply a gate to an MPS.

    Parameters
    ----------
    gate : np.ndarray
        Tensor gate.
    site : int
        First site the gate acts on.
    psi : mps
    rev : bool, optional
        True = sweep left, false = sweep right. The default is False.
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
        Updated MPS.

    """
    # Find the interaction range of the gate
    rng = gateSize(gate)
    
    # Find the product of all sites which the gate is applied too.
    prod = psi.tensors[site]
    for i in range(rng - 1):
        prod = contract(prod, psi.tensors[site+1+i], 2+i, 0)
    
    # Contract with the gate
    prod = contract(prod, gate, 1, 1)
    for i in range(rng - 1):
        prod = trace(prod, 1, 3+rng)
    prod = permute(prod, 1)
    
    # Replace the tensors
    psi.replacebond(site, prod, rev, mindim=mindim, maxdim=maxdim,
                    cutoff=cutoff, nsites=rng)
    return psi


def applyGates(gates : gateList, psi, mindim=1, maxdim=0, cutoff=0):
    """
    Apply an entire Gate List to an MPS.

    Parameters
    ----------
    gates : gateList
        List of gates to apply to the MPS.
    psi : TYPE
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
        Evolved MPS

    """
    for row in range(len(gates.gates)):
        # Move orthogonal center to the correct place
        firstSite = gates.sites[row][0]
        lastSite = gates.sites[row][-1] + gateSize(gates.gates[row][-1]) - 1
        if abs(psi.center - firstSite) < abs(psi.center - lastSite):
            rev = 0
        else:
            rev = 1
            
        # Apply gates in row
        for i in range(len(gates.gates[row])):
            # Determine the gate
            gate = i if not rev else len(gates.gates[row]) - 1 - i
            
            # Move orthogoal center
            center = gates.sites[row][gate]
            center += gateSize(gates.gates[row][gate]) - 1 if rev else 0
            psi.orthogonalize(center, mindim = mindim, maxdim = maxdim,
                              cutoff = cutoff)
            
            # Apply the gate
            applyGate(gates.gates[row][gate], gates.sites[row][gate], psi, rev,
                      mindim, maxdim, cutoff)
        
    return psi
        