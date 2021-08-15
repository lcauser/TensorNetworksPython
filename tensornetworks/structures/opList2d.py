# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.sitetypes import sitetypes 
from tensornetworks.structures.mps import mps
from tensornetworks.structures.projMPS import projMPS
import numbers

class opList2d:
    
    def __init__(self, s : sitetypes, length):
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
    
    
    def add(self, ops, sites, direction=0, coeff=1):
        if not isinstance(ops, list):
            ops = [ops]
        if not isinstance(sites, list):
            raise ValueError("The sites must be a list of two indexs to a first tensor.")
        if len(sites) != 2:
            raise ValueError("The site must be an a list of two ints.")
        
        for op in ops:
            if op not in self.sitetype.opNames:
                raise ValueError("The operator is not defined.")
            
        if not isinstance(coeff, numbers.Number):
            raise ValueError("The site must be a real or complex number.")
            
                
        # Add to the operator list
        self.ops.append(ops)
        self.sites.append(sites)
        self.coeffs.append(coeff)
        self.directions.append(direction)
        
        return self
    
    def siteIndexs(self, sites, direction):
        idxs = []
        for i in range(len(self.sites)):
            if self.sites[i] == sites and self.directions[i] == direction:
                idxs.append(i)
        return idxs

            
        
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
        
