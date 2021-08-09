# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.sitetypes1d import sitetypes1d 
import numbers

class opList:
    
    def __init__(self, s : sitetypes1d, length):
        self.length = length
        self.sitetype = s
        self.ops = []
        self.sites = []
        self.coeffs = []
        return None
    
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
            if not isinstance(site, int):
                raise ValueError("The site must be an integer.")
        
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
    
    
    def siteRange(self):
        rng = 1
        for sites in self.sites:
            rng = max(rng, max(sites)-min(sites)+1)
        return rng
    
    
    def siteIndexs(self, site):
        idxs = []
        for i in range(len(self.sites)):
            if min(self.sites[i]) == site:
                idxs.append(i)
        return idxs
    
    
    def toTensor(self, idx):
        ops = self.ops[idx]
        sites = self.sites[idx]
        rng = min(self.siteRange(), self.length-min(sites))
        
        prod = tensor((1, 1), [[1]])
        for site in range(rng):
            if min(sites)+site in sites:
                op = ops[site]
            else:
                op = "id"
            
            op = np.reshape(self.sitetype.op(op), (1, self.sitetype.dim,
                                                   self.sitetype.dim, 1))
            prod = contract(prod, op, len(np.shape(prod))-1, 0)
        prod = trace(prod, 0, len(np.shape(prod))-1)
        
        return self.coeffs[idx]*prod
        
    
    def siteTensor(self, site):
        idxs = self.siteIndexs(site)
        if len(idxs) == 0:
            return False
        ten = 0
        for idx in idxs:
            ten += self.toTensor(idx)
        
        return ten
            
            

