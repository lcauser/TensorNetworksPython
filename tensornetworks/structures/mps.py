# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *

class mps:
    
    def __init__(self, dim, length, dtype=np.complex128):
        self.dim = dim
        self.length = length
        self.dtype = dtype
        self.center = None
        
        # Create the structure of the mps
        self.createStructure()
    
    
    #def __add__(self, other):
    #    newMPS = mps(self.dim, self.length, self.dtype)
    #    for i in range(self.length):
        
    def __mul__(self, other):
        psi = copy.deepcopy(self)
        if isinstance(other, (float, complex, int, np.complex, np.float, np.int)):
            if psi.center != None:
                psi.tensors[psi.center] *= other
            else:
                psi.tensors[0] *= other
            return psi
        elif isinstance(other, mps):
            return dot(psi, other)
        
        return False
    
    def __div__(self, other):
        psi = copy.deepcopy(self)
        if isinstance(other, (float, complex, int, np.complex, np.float, np.int)):
            if psi.center != None:
                psi.tensors[psi.center] /= other
            else:
                psi.tensors[0] /= other
            return psi
        
        return False
       
    def __rmul__(self, other):
        psi = copy.deepcopy(self)
        if isinstance(other, (float, complex, int, np.complex, np.float, np.int)):
            if psi.center != None:
                psi.tensors[psi.center] *= other
            else:
                psi.tensors[0] *= other
            return psi
        elif isinstance(other, mps):
            return np.conj(dot(psi, other))
        
        return False
            
            
    
    def createStructure(self):
        tensors = []
        for i in range(self.length):
            tensors.append(tensor((1, self.dim, 1)))
        self.tensors = tensors
        return self
    
    
    def bondDim(self, idx):
        return np.shape(self.tensors[idx])[2]
    
    
    def maxBondDim(self):
        dim = 1
        for idx in range(self.length):
            dim = max(dim, self.bondDim(idx))
        return dim
            
    
    def moveLeft(self, idx, mindim=1, maxdim=0, cutoff=0):
        # Get the SVD at the site
        U, S, V = svd(self.tensors[idx], 0, mindim, maxdim, cutoff)
        
        # Get the site to the left
        A = self.tensors[idx-1]
        A = contract(A, V, 2, 1)
        A = contract(A, S, 2, 1)
        
        self.tensors[idx-1] = A
        self.tensors[idx] = U
        
        return self
    
    def moveRight(self, idx, mindim=1, maxdim=0, cutoff=0):
        # Get the SVD at the site
        U, S, V = svd(self.tensors[idx], 2, mindim, maxdim, cutoff)
        
        # Get the site to the right
        A = self.tensors[idx+1]
        S = contract(S, V, 1, 0)
        A = contract(S, A, 1, 0)
        
        self.tensors[idx+1] = A
        self.tensors[idx] = U
        
        return self
    
    
    def orthogonalize(self, idx, mindim=1, maxdim=0, cutoff=0):
        # Check to see if already in a canonical form
        if self.center == None:
            for i in range(self.length-1):
                self.moveLeft(self.length-1-i, mindim, maxdim, cutoff)
            self.center = 0
        
        
        if idx < self.center:
            for i in range(self.center-idx):
                self.moveLeft(self.center-i, mindim, maxdim, cutoff)
            self.center = idx
        
        if idx > self.center:
            for i in range(idx-self.center):
                self.moveRight(self.center+i, mindim, maxdim, cutoff)
            self.center = idx
        
        return self
    
    
    def norm(self):
        if self.center == None:
            self.orthogonalize(0)
        
        return norm(self.tensors[self.center])
    
    
    def normalize(self):
        norm = self.norm()
        self.tensors[self.center] *= (1/norm)
    
    
    def truncate(self, mindim=1, maxdim=0, cutoff=0):
        if self.center != 0 and self.center != self.length-1:
            self.orthogonalize(self.length-1)
        
        self.orthogonalize(self.length-1 - self.center, mindim, maxdim, cutoff)
        return self
    
    
    """
        Update this to support n-sites.
    """
    def replacebond(self, site, A, rev=False, mindim=1, maxdim=0, cutoff=0):
        # Group the indices together
        dims = np.shape(A)
        A, C1 = combineIdxs(A, [0, 1])
        A, C2 = combineIdxs(A, [0, 1])
        
        # Do a SVD
        U, S, V = svd(A, 1, mindim, maxdim, cutoff)
        #U = permute(U, 0, -1)
        #U = uncombineIdxs(U, C1)
        #V = uncombineIdxs(V, C2)
        #V = permute(V, 2, 0)
        U = reshape(U, (dims[0], dims[1], np.shape(S)[0]))
        V = reshape(V, (np.shape(S)[1], dims[2], dims[3]))
        
        # Absorb singular values
        if rev:
            U = contract(U, S, 2, 0)
            self.center = site
        else:
            V = contract(S, V, 1, 0)
            self.center = site+1
        
        self.tensors[site] = U
        self.tensors[site+1] = V
        
        return self
        




def randomMPS(dim, length, bondDim=1, normalize=True, dtype=np.complex128):
    # Create MPS
    psi = mps(dim, length, dtype)
    
    maxdims = [1]
    for i in range(length-1):
        maxdims.append(dim**(min(i+1, length-i-1)))
    maxdims.append(1)
    
    # Allocate random tensors
    maxlinkdim = dim
    for i in range(length):
        dims = (min(bondDim, maxdims[i]), dim, min(bondDim, maxdims[i+1]))        
        psi.tensors[i] = randomTensor(dims, dtype)
    
    # Move orthogonal centre to first site and normalize
    psi.orthogonalize(0)
    if normalize:
        psi *= 1 / psi.norm()

    
    return psi


def productMPS(dim, length, A, dtype=np.complex128):
    # Create MPS
    psi = mps(dim, length, dtype)
    A = np.reshape(A, (1, np.size(A), 1))
    A = tensor(np.shape(A), A, dtype)
    for i in range(length):
        psi.tensors[i] = copy.deepcopy(A)
    
    return psi


def dot(psi : mps, phi : mps):
    # Initilize at first site
    A1 = psi.tensors[0]
    A2 = dag(phi.tensors[0])
    prod = contract(A1, A2, 1, 1)
    prod = permute(prod, 2, 1)
    
    # Loop through contracting
    for i in range(psi.length - 1):
        A1 = psi.tensors[i+1]
        A2 = dag(phi.tensors[i+1])
        prod = contract(prod, A1, 2, 0)
        prod = contract(prod, A2, 2, 0)
        prod = trace(prod, 2, 4)
    
    prod = trace(prod, 0, 1)
    prod = trace(prod, 0, 1)
    
    return np.asscalar(prod)
    