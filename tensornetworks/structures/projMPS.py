"""
    Convinient to many algorithms is to project a dot product between two
    MPS onto just a single (or multiple) site(s) of an MPS, allowing for
    optimizations etc. This deals with this efficiently.
"""

# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.mps import *
from scipy.sparse.linalg import LinearOperator

class projMPS(LinearOperator):
    
    def __init__(self, phi : mps, psi : mps, center=0, nsites=2, squared=False):
        """
        Project the dot product of two MPS onto the sites of an MPS.

        Parameters
        ----------
        phi : mps
            MPS which is the dot product.
        psi : mps
            MPS which will be projected onto.
        center : int, optional
            Build the projMPS onto the center site. The default is 0.
        nsites : int, optional
            Number of sites to project onto. The default is 2.
        squared : bool, optional
            Calculate the norm of the dot?. The default is False.

        """
        # Ensure the two MPS have the same properties
        if psi.dim != phi.dim or psi.length != phi.length:
            raise ValueError("MPS properties must match!")
        
        # Save references to the class
        self.psi = psi
        self.phi = phi
        self.nsites = nsites
        self.squared = squared
        self.dtype = self.psi.dtype
        self.rev = False
        
        # Create the block structure
        self.blocks = [0] * self.psi.length
        self.center = 0
        self.center = self.psi.length - 1
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
        A1 = dag(self.phi.tensors[idx])
        A2 = self.psi.tensors[idx]
        if idx == 0:
            prod = ones((1, 1))
        else:
            prod = self.blocks[idx-1]
        
        # Contract sites onto the block
        prod = contract(prod, A1, 0, 0)
        prod = contract(prod, A2, 0, 0)
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
        A1 = dag(self.phi.tensors[idx])
        A2 = self.psi.tensors[idx]
        if idx == self.psi.length-1:
            prod = ones((1, 1))
        else:
            prod = self.blocks[idx+1]
        
        # Contract sites onto the block
        prod = contract(A2, prod, 2, 1)
        prod = contract(A1, prod, 2, 2)
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
            left = ones((1, 1))
        else:
            left = self.blocks[site1-1]
            
        if site1 + self.nsites == self.psi.length:
            right = ones((1, 1))
        else:
            right = self.blocks[site1+self.nsites]
        
        # Reverse block and contract with tensors
        prod = permute(left, 0)
        for i in range(self.nsites):
            A = dag(self.phi.tensors[site1+i])
            prod = contract(prod, A, len(shape(prod))-1, 0)
        prod = dag(contract(prod, right, len(shape(prod))-1, 0))
        
        # Find the contraction with x
        if self.squared:
            prod2 = contract(left, y, 1, 0)
            prod2 = permute(prod2, 0)
            for i in range(self.nsites):
                A = dag(self.phi.tensors[site1+i])
                prod2 = contract(prod2, A, len(shape(prod2))-1, 0)
                prod2 = trace(prod2, 0, len(shape(prod2))-2)
            prod2 = contract(prod2, right, 0, 1)
            prod2 = trace(prod2, 0, 1)
            prod *= prod2
        
        return prod.flatten()            
        
    
    
    def opShape(self):
        shape = []
        site = self.center - self.rev*(self.nsites-1)
        shape.append(np.shape(self.psi.tensors[site])[0])
        for i in range(self.nsites):
            shape.append(self.psi.dim)
        shape.append(np.shape(self.psi.tensors[site+self.nsites-1])[2])
        
        return (np.prod(shape),np.prod(shape))
            