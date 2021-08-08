# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.mps import *
from scipy.sparse.linalg import LinearOperator

class projMPS(LinearOperator):
    
    def __init__(self, psi, phi, center=0, nsites=2, squared=False):
        if psi.dim != phi.dim or psi.length != phi.length:
            raise("MPS properties must match!")
            return False       
        self.psi = psi
        self.phi = phi
        self.nsites = nsites
        self.squared = squared
        self.dtype = self.psi.dtype
        self.blocks = [0] * self.psi.length
        self.center = 0
        self.rev = False
        self.moveCenter(self.psi.length - 1)
        self.moveCenter(0)
        self.moveCenter(center)
        self.shape = self.opShape()
        
    
    def buildLeft(self, idx):
        A1 = self.psi.tensors[idx]
        A2 = dag(self.phi.tensors[idx])
        if idx == 0:
            prod = contract(A1, A2, 1, 1)
            prod = permute(prod, 2, 1)
        else:
            prod = self.blocks[idx-1]
            prod = contract(prod, A1, 2, 0)
            prod = contract(prod, A2, 2, 0)
            prod = trace(prod, 2, 4)
        self.blocks[idx] = prod
        return True
    
    
    def buildRight(self, idx):
        A1 = self.psi.tensors[idx]
        A2 = dag(self.phi.tensors[idx])
        if idx == self.psi.length-1:
            prod = contract(A1, A2, 1, 1)
            prod = permute(prod, 2, 1)
        else:
            prod = self.blocks[idx+1]
            prod = contract(A2, prod, 2, 1)
            prod = contract(A1, prod, 2, 2)
            prod = trace(prod, 1, 3)
        self.blocks[idx] = prod
        return True
    
    
    def moveCenter(self, idx):        
        if idx < self.center:
            for i in range(self.center-idx):
                self.buildRight(self.center-i)
            self.center = idx
        
        if idx > self.center:
            for i in range(idx-self.center):
                self.buildLeft(self.center+i)
            self.center = idx
        
        return True

    
    def _matvec(self, x):
        # Retrieve tensors at the two sites
        site1 = self.center if not self.rev else self.center - 1
        site2 = self.center+1 if not self.rev else self.center
        A1 = dag(self.phi.tensors[site1])
        A2 = dag(self.phi.tensors[site2])
        
        # Retrieve edge blocks
        if site1 == 0:
            left = ones((1, 1, 1, 1))
        else:
            left = self.blocks[site1-1]
        if site2 == self.psi.length - 1:
            right = ones((1, 1, 1, 1))
        else:
            right = self.blocks[site2+1]
        
        prod = contract(left, A1, 3, 0)
        prod = contract(prod, A2, 4, 0)
        prod = contract(prod, right, 5, 1)
        prod = trace(prod, 0, 6)
        prod = trace(prod, 0, 5)
        prod = prod.storage.flatten()
        if self.squared == True:
            prod = np.outer(np.conj(prod), prod)
        return prod
    
    
    def opShape(self):
        shape = []
        shape.append(np.shape(self.psi.tensors[self.center])[0])
        for i in range(self.nsites):
            shape.append(self.psi.dim)
        shape.append(np.shape(self.psi.tensors[self.center+self.nsites-1])[2])
        
        return (np.prod(shape),np.prod(shape))
            