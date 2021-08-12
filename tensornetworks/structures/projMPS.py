# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.mps import *
from scipy.sparse.linalg import LinearOperator

class projMPS(LinearOperator):
    
    def __init__(self, phi, psi, center=0, nsites=2, squared=False):
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
        self.center = self.psi.length - 1
        self.moveCenter(0)
        self.moveCenter(center)
        self.shape = lambda: self.opShape()
        
    
    def buildLeft(self, idx):
        A1 = dag(self.phi.tensors[idx])
        A2 = self.psi.tensors[idx]
        if idx == 0:
            prod = ones((1, 1))
        else:
            prod = self.blocks[idx-1]
        
        prod = contract(prod, A1, 0, 0)
        prod = contract(prod, A2, 0, 0)
        prod = trace(prod, 0, 2)
        
        self.blocks[idx] = prod
        return True
    
    
    def buildRight(self, idx):
        A1 = dag(self.phi.tensors[idx])
        A2 = self.psi.tensors[idx]
        if idx == self.psi.length-1:
            prod = ones((1, 1))
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
        #print('no')
        # Retrieve tensors at the two sites
        site1 = self.center if not self.rev else self.center - 1
        site2 = self.center+1 if not self.rev else self.center
        A1 = dag(self.phi.tensors[site1])
        A2 = dag(self.phi.tensors[site2])
        
        # Retrieve edge blocks
        if site1 == 0:
            left = ones((1, 1))
        else:
            left = self.blocks[site1-1]
        if site2 == self.psi.length - 1:
            right = ones((1, 1))
        else:
            right = self.blocks[site2+1]
        

        prod = contract(left, A1, 0, 0)
        prod = contract(prod, A2, 2, 0)
        prod2 = dag(contract(prod, right, 3, 0))
        
        if self.squared == True:
            D1 = np.shape(self.psi.tensors[site1])[0]
            D2 = np.shape(self.psi.tensors[site2])[2]
            d = self.psi.dim
            y = dag(np.reshape(x, (D1, d, d, D2)))
            #print(y)
            prod = contract(prod, y, 0, 0)
            prod = trace(prod,0, 3)
            prod = trace(prod, 0, 2)
            prod = contract(prod, right, 0, 0)
            prod = trace(prod, 0, 1)
            prod2 = prod*prod2
            #print(prod)
            #print(prod2)
        
        return prod2.flatten()
    
    
    def opShape(self):
        shape = []
        site = self.center - self.rev*(self.nsites-1)
        shape.append(np.shape(self.psi.tensors[site])[0])
        for i in range(self.nsites):
            shape.append(self.psi.dim)
        shape.append(np.shape(self.psi.tensors[site+self.nsites-1])[2])
        
        return (np.prod(shape),np.prod(shape))
            