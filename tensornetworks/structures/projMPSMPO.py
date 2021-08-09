# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.mps import *
from scipy.sparse.linalg import LinearOperator

class projMPSMPO(LinearOperator):
    
    def __init__(self, H, psi, center=0, nsites=2):
        if psi.dim != H.dim or psi.length != H.length:
            raise("MPS and MPO properties must match!")
            return False       
        self.psi = psi
        self.O = H
        self.nsites = nsites
        self.dtype = self.psi.dtype
        self.blocks = [0] * self.psi.length
        self.center = 0
        self.rev = False
        self.moveCenter(self.psi.length - 1)
        self.moveCenter(0)
        self.moveCenter(center)
        self.shape = lambda: self.opShape()
        
    
    def buildLeft(self, idx):
        # Get the previous block
        if idx == 0:
            prod = np.ones((1, 1, 1))
        else:
            prod = self.blocks[idx-1]
        
        # Get the tensors
        A = self.psi.tensors[idx]
        M = self.O.tensors[idx]
        
        # Expand the block
        prod = contract(prod, dag(A), 0, 0)
        prod = contract(prod, M, 0, 0)
        prod = trace(prod, 1, 3)
        prod = contract(prod, A, 0, 0)
        prod = trace(prod, 1, 3)
        
        # Update the block
        self.blocks[idx] = prod
        return True
    
    
    def buildRight(self, idx):
        # Get the previous block
        if idx == self.psi.length - 1:
            prod = np.ones((1, 1, 1))
        else:
            prod = self.blocks[idx+1]
        
        # Get the tensors
        A = self.psi.tensors[idx]
        M = self.O.tensors[idx]
        
        # Expand the block
        prod = contract(dag(A), prod, 2, 0)
        prod = contract(M, prod, 3, 2)
        prod = trace(prod, 1, 4)
        prod = contract(A, prod, 2, 3)
        prod = trace(prod, 1, 3)
        prod = permute(prod, 1)
        prod = permute(prod, 0)
        
        # Update the block
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
        #print('yes')
        # Determine the sites
        site1 = self.center if not self.rev else self.center - 1
        site2 = self.center + 1 if not self.rev else self.center
        
        # Fetch the tensors
        M1 = self.O.tensors[site1]
        M2 = self.O.tensors[site2]
        
        # Reshape the tensor into the correct form
        D1 = np.shape(self.psi.tensors[site1])[0]
        D2 = np.shape(self.psi.tensors[site2])[2]
        d = self.psi.dim
        y = np.reshape(x, (D1, d, d, D2))
        
        # Retrieve edge blocks
        if site1 == 0:
            left = ones((1, 1, 1))
        else:
            left = self.blocks[site1-1]
        if site2 == self.psi.length - 1:
            right = ones((1, 1, 1))
        else:
            right = self.blocks[site2+1]
            
        # Do the contractions
        prod = contract(left, M1, 1, 0)
        prod = contract(prod, y, 1, 0)
        prod = trace(prod, 2, 4)
        prod = contract(prod, M2, 2, 0)
        prod = trace(prod, 2, 5)
        prod = contract(prod, right, 2, 2)
        prod = trace(prod, 3, 5)
        
        return prod.flatten()
    
    
    def opShape(self):
        shape = []
        site = self.center - self.rev*(self.nsites-1)
        shape.append(np.shape(self.psi.tensors[site])[0])
        for i in range(self.nsites):
            shape.append(self.psi.dim)
        shape.append(np.shape(self.psi.tensors[site+self.nsites-1])[2])
        
        return (np.prod(shape),np.prod(shape))
            