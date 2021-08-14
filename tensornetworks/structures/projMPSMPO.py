"""
    Convinient to many algorithms is to project the expectation between a MPS
    and a MPO onto just a single (or multiple) site(s) of an MPS, allowing for
    optimizations etc. This deals with this efficiently.
"""

# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.mps import *
from scipy.sparse.linalg import LinearOperator

class projMPSMPO(LinearOperator):
    
    def __init__(self, H, psi, center=0, nsites=2):
        """
        Project the expectation of a MPS and MPO onto the sites of an MPS.

        Parameters
        ----------
        H : mpo
            MPO in the expectation
        psi : mps
            MPS which will be projected onto.
        center : int, optional
            Build the projMPS onto the center site. The default is 0.
        nsites : int, optional
            Number of sites to project onto. The default is 2.

        """
        # Check MPS and MPO have same properties
        if psi.dim != H.dim or psi.length != H.length:
            raise TypeError("MPS and MPO properties must match!")
        
        # Store references
        self.psi = psi
        self.O = H
        self.nsites = nsites
        self.dtype = self.psi.dtype
        self.rev = False
        
        # Create block structure
        self.blocks = [0] * self.psi.length
        self.center = 0
        self.moveCenter(self.psi.length - 1)
        self.moveCenter(0)
        self.moveCenter(center)
        
        # Shape of projMPSMPO acting on sites
        self.shape = lambda: self.opShape()
        
    
    def buildLeft(self, idx):
        """
        Build up the blocks from the left.

        Parameters
        ----------
        idx : int
            Site index to build to.

        """
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
        return None
    
    
    def buildRight(self, idx):
        """
        Build up the blocks from the right.

        Parameters
        ----------
        idx : int
            Site index to build to.

        """
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

    
    def _matvec(self, x):
        # Determine the sites
        site1 = self.center if not self.rev else self.center - self.nsites + 1
        
        # Reshape the tensor into the correct form
        dims = [shape(self.psi.tensors[site1])[0]]
        for i in range(self.nsites):
            dims.append(self.psi.dim)
        dims.append(shape(self.psi.tensors[site1+self.nsites-1])[2])
        y = reshape(x, dims)
        
        # Retrieve the edge blocks
        if site1 == 0:
            left = ones((1, 1, 1))
        else:
            left = self.blocks[site1-1]
            
        if site1 + self.nsites == self.psi.length:
            right = ones((1, 1, 1))
        else:
            right = self.blocks[site1+self.nsites]
        
        # Contract the tensor with the left block
        prod = contract(left, y, 2, 0)
        prod = permute(prod, 1)
        
        # Loop through nsites and contract
        for i in range(self.nsites):
            # Fetch the site tensor
            M = self.O.tensors[site1+i]
            
            # Contract
            prod = contract(prod, M, len(shape(prod))-1, 0)
            prod = trace(prod, 1, len(shape(prod))-2)
        
        # Contract with right block
        #print(shape(prod))
        #print(shape(right))
        prod = contract(prod, right, 1, 2)
        prod = trace(prod, len(shape(prod))-3, len(shape(prod))-1)
        
        return prod.flatten()
    
    
    def opShape(self):
        shape = []
        site = self.center - self.rev*(self.nsites-1)
        shape.append(np.shape(self.psi.tensors[site])[0])
        for i in range(self.nsites):
            shape.append(self.psi.dim)
        shape.append(np.shape(self.psi.tensors[site+self.nsites-1])[2])
        
        return (np.prod(shape),np.prod(shape))
            