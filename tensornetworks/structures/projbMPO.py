"""
    Constructs the projection of a dot of two bMPOs onto one of the tensors
    in a bMPO.
"""

# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.mpo import mpo
from scipy.sparse.linalg import LinearOperator

class projbMPO():
    
    def __init__(self, O : mpo, P : mpo, center=0):
        """
        Project the dot product of two bMPOs onto the site of a bMPO.

        Parameters
        ----------
        O : mpo
            bMPO which is the dot product.
        P : mpo
            bMPO which will be projected onto.
        center : int, optional
            Build the projbMPO onto the center site. The default is 0.

        """
        
        # Save references to the class
        self.O = O
        self.P = P
        self.dtype = self.P.dtype
        #self.rev = False
        
        # Create the block structure
        self.blocks = [0] * self.P.length
        self.center = 0
        self.center = self.P.length - 1
        self.moveCenter(0)
        self.moveCenter(center)
        
        # Define the shape of the class acting on the sites.
        self.shape = lambda: self.opShape()
        
    
    def buildLeft(self, idx):
        """
        Build up the blocks from the left.

        Parameters
        ----------
        idx : int
            Site index to build to.

        """
        # Fetch the site tensors and previous block
        A1 = dag(self.O.tensors[idx])
        A2 = self.P.tensors[idx]
        if idx == 0:
            prod = ones((1, 1))
        else:
            prod = self.blocks[idx-1]
        
        # Contract sites onto the block
        prod = contract(prod, A1, 0, 0)
        prod = contract(prod, A2, 0, 0)
        prod = trace(prod, 0, 3)
        prod = trace(prod, 0, 2)
        
        # Store it
        self.blocks[idx] = prod
        return None
    
    
    def buildRight(self, idx):
        """
        Build up the blocks from the right.

        Parameters
        ----------
        idx : int
            Site index to build to.

        """
        # Fetch the site tensors and previous block
        A1 = dag(self.O.tensors[idx])
        A2 = self.P.tensors[idx]
        if idx == self.P.length-1:
            prod = ones((1, 1))
        else:
            prod = self.blocks[idx+1]
        
        # Contract sites onto the block
        prod = contract(A2, prod, 3, 1)
        prod = contract(A1, prod, 3, 3)
        prod = trace(prod, 1, 4)
        prod = trace(prod, 1, 3)
        
        # Store it
        self.blocks[idx] = prod
        return None
    
    
    def moveCenter(self, idx):        
        """
        Move the center of the projection by building blocks.

        Parameters
        ----------
        idx : int
            Site index to build to.
        """
        # Build from the right
        if idx < self.center:
            for i in range(self.center-idx):
                self.buildRight(self.center-i)
            self.center = idx
        
        # Build from the left
        if idx > self.center:
            for i in range(idx-self.center):
                self.buildLeft(self.center+i)
            self.center = idx
        
        return None
    
    
    def calculate(self):
        """ Calculates the projected overlap. """
        
        # Determine the site
        site = self.center
        
        # Retrieve site blocks
        O = dag(self.O.tensors[site])
        P = self.P.tensors[site]
        
        # Retrieve the edge blocks
        if site == 0:
            left = ones((1, 1))
        else:
            left = self.blocks[site-1]
            
        if site == self.P.length - 1:
            right = ones((1, 1))
        else:
            right = self.blocks[site+1]
                                
        # Contract
        prod = contract(left, O, 0, 0)
        prod = contract(prod, P, 0, 0)
        prod = trace(prod, 0, 3)
        prod = trace(prod, 0, 2)
        prod = contract(prod, right, 0, 0)
        prod = trace(prod, 0, 1)
        
        return np.asscalar(prod)

    
    def vector(self):
        # Determine the site
        site = self.center
        
        # Retrieve the edge blocks
        if site == 0:
            left = ones((1, 1))
        else:
            left = self.blocks[site-1]
            
        if site == self.P.length - 1:
            right = ones((1, 1))
        else:
            right = self.blocks[site+1]
            
        # Contract the blocks with the O tensor
        O = dag(self.O.tensors[site])
        prod = contract(left, O, 0, 0)
        prod = contract(prod, right, 3, 0)
        
        return prod
              
    
    def opShape(self):
        return shape(self.P.tensors[self.center])
            