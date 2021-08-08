# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.mps import mps

class mpo:
    
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
            tensors.append(tensor((1, self.dim, self.dim, 1)))
        self.tensors = tensors
        return self
    
    
    def bondDim(self, idx):
        return self.np.shape(tensors[idx])[3]
    
    
    def maxBondDim(self):
        dim = 1
        for idx in range(self.length):
            dim = max(dim, self.bondDim(idx))
        return dim
        
    
def uniformMPO(dim, length, M, dtype=np.float64):
    bonddim = np.shape(M)[0]
    O = mpo(dim, length, dtype)
    O.tensors[0] = M[bonddim-1:bonddim, :, :, :]
    for i in range(length-2):
        O.tensors[1+i] = M
    O.tensors[length-1] = M[:, :, :, 0:1]
    return O


def applyMPO(O : mpo, psi : mps):
    psi2 = copy.deepcopy(psi)
    for i in range(O.length):
        A = contract(O.tensors[i], psi.tensors[i], 2, 1)
        A, c = combineIdxs(A, [0, 3])
        A, c = combineIdxs(A, [1, 2])
        A = permute(A, 0, 1)
        psi2.tensors[i] = A
    
    return psi2


def inner(psi1 : mps, O : mpo, psi2 : mps):
    # Make initial product tensor
    prod = np.ones((1, 1, 1))
    
    for i in range(psi1.length):
        # Fetch site tensors
        M1 = dag(psi1.tensors[i])
        M2 = psi2.tensors[i]
        A = O.tensors[i]
        
        # Contractions
        prod = contract(prod, M1, 0, 0)
        prod = contract(prod, A, 0, 0)
        prod = trace(prod, 1, 3)
        prod = contract(prod, M2, 0, 0)
        prod = trace(prod, 1, 3)
    
    return np.asscalar(prod)
    
    
        