# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.sitetypes1d import sitetypes1d 
from tensornetworks.structures.mps import mps
from tensornetworks.structures.projMPS import projMPS
from tensornetworks.structures.gateList import gateList
import numbers

class opList2d:
    
    def __init__(self, s : sitetypes1d, length):
        if not isinstance(length, list):
            length = [length, length]
        self.length = length
        self.sitetype = s
        self.ops = []
        self.sites = []
        self.directions = []
        self.coeffs = []
        return None
    
    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            opList = copy.deepcopy(self)
            for idx in range(len(opList.coeffs)):
                opList.coeffs[idx] *= other
            opList.sitetype = self.sitetype
        return opList
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    
    def __add__(self, other):
        if isinstance(other, opList):
            return addOpLists(self, other)
        raise ValueError("Type error.")
        
        
    def __radd__(self, other):
        return self.__add__(other)
    
    
    def add(self, ops, sites, coeff=1):
        if not isinstance(ops, list):
            ops = [ops]
        if not isinstance(sites, list):
            sites = [sites]
        
        # Check the length of ops and sites are the same
        if len(ops) != len(sites):
            raise ValueError("The operator and site list must be of equal size.")
        
        for op in ops:
            if op not in self.sitetype.opNames:
                raise ValueError("The operator is not defined.")
        
        for site in sites:
            if not isinstance(site, list):
                raise ValueError("The site must be an a list of two ints.")
            
        
        if not isinstance(coeff, numbers.Number):
            raise ValueError("The site must be a real or complex number.")
            
        
        # Sort the operator and site list
        ops = [op for _, op in sorted(zip(sites, ops))]
        sites = sorted(sites)
        
        # Add to the operator list
        self.ops.append(ops)
        self.sites.append(sites)
        self.coeffs.append(coeff)
        
        return self
    
    

            
        
def applyOp(psi, sitetype, ops, sites, coeff):    
    psi = copy.deepcopy(psi)
    # Loop through each operator
    for i in range(len(ops)):
        # Fetch site tensors
        A = psi.tensors[sites[i][0]][sites[i][1]]
        O = sitetype.op(ops[i])
        
        # Apply operator and restore tensor
        A = contract(A, O, 4, 1)
        psi.tensors[sites[i][0]][sites[i][1]] = coeff*A
    
    return psi
        

def trotterize(ops : opList2d, timestep : float, order : int = 1):
    # Create the gatelist and find the interaction range
    gl = gateList(ops.length)

    # Create horizontal even bonds
    for i in range(ops.length[0]):
        

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