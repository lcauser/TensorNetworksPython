# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.sitetypes1d import sitetypes1d
from tensornetworks.structures.mps import mps

class peps:
    
    def __init__(self, dim : int, length, collapsed = False,
                 dtype=np.complex128):
        # Save the phyiscal dimension
        self.dim = dim
        
        # Save the length
        if not isinstance(length, list):
            length = [length, length]
        if len(length) != 2:
            raise ValueError("PEPS networks must be two dimensional.")
        self.length = length
        
        # Save the dtype
        self.dtype = dtype
        
        # Is the PEPs collapsed i.e. are the physical dimensions chosen?
        self.collapsed = collapsed
        
        # Create the structure of the mps
        self.createStructure()
    
    

    def __mul__(self, other):
        
        return False
    
    def __div__(self, other):
        
        return False
       
    def __rmul__(self, other):
            
        return False
            
            
    
    def createStructure(self):
        tensors = []
        for i in range(self.length[0]):
            tensorsX = []
            for j in range(self.length[1]):
                if not self.collapsed:
                    tensorsX.append(tensor((1, 1, 1, 1, self.dim)))
                else:
                    tensorsX.append(tensor((1, 1, 1, 1)))
            tensors.append(tensorsX)
        self.tensors = tensors
        return self
    
    
    def bondDim(self, idx1, idx2, vertical=1):
        if vertical:
            return np.shape(self.tensors[idx1][idx2])[2]
        else:
            return np.shape(self.tensors[idx1][idx2])[3]
    
    
    def maxBondDim(self):
        maxdim = 0
        for i in range(self.length[0]):
            for j in range(self.length[1]):
                maxdim = max(maxdim, self.bondDim(i, j, 0))
                maxdim = max(maxdim, self.bondDim(i, j, 1))
        return maxdim
            
    
    
    def norm(self, chi=0):
        return dotPEPS(self, self, chi=chi)
    
    
    def normalize(self, chi=0):
        norm = np.sqrt(self.norm(chi=chi))**(1/(self.length[0]*self.length[1]))
        for i in range(self.length[0]):
            for j in range(self.length[1]):
                self.tensors[i][j] /= norm
        return None
    
    
    def truncate(self, mindim=1, maxdim=0, cutoff=0):
        # Loop through each bond pair
        for i in range(self.length[0]):
            for j in range(self.length[1]):
                # Replace horizontal bond
                if j != self.length[1] - 1:
                    # Replace horizontal bond
                    A1 = self.tensors[i][j]
                    A2 = self.tensors[i][j+1]
                    A = contract(A1, A2, 3, 0)
                    dims = np.shape(A)
                    A, cmb1 = combineIdxs(A, [0, 1, 2])
                    A, cmb2 = combineIdxs(A, [0, 1, 2])
                    U, S, V = svd(A, 1, mindim, min(maxdim, np.shape(A1)[3]), cutoff)
                    V = contract(S, V, 1, 0)
                    U = np.reshape(U, (dims[0], dims[1], dims[2], np.shape(S)[0]))
                    V = np.reshape(V, (np.shape(S)[1], dims[3], dims[4], dims[5]))
                    self.tensors[i][j] = U
                    self.tensors[i][j+1] = V
                
                # Replace vertical bond
                if i != self.length[0] - 1:
                    A1 = self.tensors[i][j]
                    A2 = self.tensors[i+1][j]
                    A = contract(A1, A2, 2, 1)
                    dims = np.shape(A)
                    A, cmb1 = combineIdxs(A, [0, 1, 2])
                    A, cmb2 = combineIdxs(A, [0, 1, 2])
                    U, S, V = svd(A, 1, mindim, min(maxdim, np.shape(A1)[2]), cutoff)
                    V = contract(S, V, 1, 0)
                    U = np.reshape(U, (dims[0], dims[1], np.shape(S)[0], dims[2]))
                    V = np.reshape(V, (dims[3], np.shape(S)[1], dims[4], dims[5]))
                    self.tensors[i][j] = U
                    self.tensors[i+1][j] = V
                
        return self
    
    

    def applyGate(self, gate, sites, mindim=1, maxdim=0, cutoff=10**-12,
                  normalize=False):
        # Ensure sites are neighbouring and ascending order
        site1 = sites[0]
        site2 = sites[1]
        if site2[0] == site1[0] + 1 and site1[1] == site2[1]:
            direction = 1
        elif site2[1] == site1[1] + 1 and site1[0] == site2[0]:
            direction = 0
        else:
            raise ValueError("Sites must neighbouring and in ascending order.")
            
        # Check that the gate is of right dimensions
        if np.shape(gate) != (self.dim, self.dim, self.dim, self.dim):
            raise ValueError("The gate must be the correct dimensions.")
        
        # Fetch the sites
        A1 = self.tensors[site1[0]][site1[1]]
        A2 = self.tensors[site2[0]][site2[1]]
        
        # Contract
        if direction == 0:
            # Contract both with gates
            prod = contract(A1, gate, 4, 1)
            prod = contract(prod, A2, 3, 0)
            prod = trace(prod, 5, 9)
            prod = permute(prod, 4)
            
            # Combine indices and apply SVD
            prod, cmb = combineIdxs(prod, [0, 1, 2, 3])
            prod, cmb = combineIdxs(prod, [0, 1, 2, 3])
            U, S, V = svd(prod, mindim=mindim, maxdim=maxdim, cutoff=cutoff)
            
            if normalize:
                S = S / np.sqrt(np.sum(S))
            
            # Reshape into tensors
            U = permute(U, 1, 0)
            U = np.reshape(U, (np.shape(S)[0], np.shape(A1)[0], np.shape(A1)[1],
                                       np.shape(A1)[2], self.dim))
            U = permute(U, 0, 3)
            V = contract(S, V, 1, 0)
            V = np.reshape(V, (np.shape(S)[1], np.shape(A2)[1], np.shape(A2)[2],
                               np.shape(A2)[3], np.shape(A2)[4]))
            
            # Replace tensors
            self.tensors[site1[0]][site1[1]] = U
            self.tensors[site2[0]][site2[1]] = V
        else:
            # Contract both with gates
            prod = contract(A1, gate, 4, 1)
            prod = contract(prod, A2, 2, 1)
            prod = trace(prod, 5, 9)
            prod = permute(prod, 4)
            
            # Combine indices and apply SVD
            prod, cmb = combineIdxs(prod, [0, 1, 2, 3])
            prod, cmb = combineIdxs(prod, [0, 1, 2, 3])
            U, S, V = svd(prod, mindim=mindim, maxdim=maxdim, cutoff=cutoff)
            
            if normalize:
                S = S / np.sqrt(np.sum(S))
            
            # Reshape into tensors
            U = permute(U, 1, 0)
            U = np.reshape(U, (np.shape(S)[0], np.shape(A1)[0], np.shape(A1)[1],
                                       np.shape(A1)[3], self.dim))
            U = permute(U, 0, 2)
            V = contract(S, V, 1, 0)
            V = np.reshape(V, (np.shape(S)[1], np.shape(A2)[0], np.shape(A2)[2],
                               np.shape(A2)[3], np.shape(A2)[4]))
            V = permute(V, 0, 1)
            
            # Replace tensors
            self.tensors[site1[0]][site1[1]] = U
            self.tensors[site2[0]][site2[1]] = V
        
        return self
        
        
        


def randomPEPS(dim, length, bondDim = 1, dtype=np.complex128):
    # Create an empty PEPS
    psi = peps(dim, length, dtype)
    
    # Loop through each tensor
    for i in range(psi.length[0]):
        for j in range(psi.length[1]):
            # Find the appropiate size
            Dleft = 1 if j == 0 else bondDim
            Dright = 1 if j == psi.length[1] - 1 else bondDim
            Dup = 1 if i == 0 else bondDim
            Ddown = 1 if i == psi.length[0] - 1 else bondDim
            
            # Make a random tensor
            A = randomTensor((Dleft, Dup, Ddown, Dright, psi.dim), dtype)
            psi.tensors[i][j] = A
    return psi


def meanfieldPEPS(dim, length, A, dtype=np.complex128):
    # Create an empty PEPS
    psi = peps(dim, length, dtype)
    
    # Reshape the tensor
    A = np.reshape(A, (1, 1, 1, 1, dim))
    A = tensor(np.shape(A), A)
    
    # Add the tensor to the PEPS
    for i in range(psi.length[0]):
        for j in range(psi.length[1]):
            psi.tensors[i][j] = A
    return psi


def productPEPS(s : sitetypes1d, states):
    # Create an empty PEPS
    psi = peps(s.dim, [len(states), len(states[0])])
    
    # Loop through each tensor
    for i in range(psi.length[0]):
        for j in range(psi.length[1]):
            psi.tensors[i][j][0, 0, 0, 0, :] = s.state(states[i][j])
    
    return psi


def dotPEPS(psi1: peps, psi2 : peps, chi = None, cutoff=0):
    # Determine the auxillary bond dimension
    if chi == None or chi == 0:
        chi = 5*max(psi1.maxBondDim(), psi2.maxBondDim())**2
        chi = max(chi, 100)
    
    # Check that psi1 and psi2 describe the same system
    if psi1.dim != psi2.dim:
        raise ValueError("PEPS networks must have the same physical dimension.")
    if psi1.length != psi2.length:
        raise ValueError("PEPS networks must have the same geometry.")
    
    # Collapse the dot product into a PEPS and truncate
    psiCollapsed = peps(psi1.dim, psi1.length, True)
    for i in range(psi1.length[0]):
        for j in range(psi2.length[1]):
            # Fetch the tensors
            A1 = dag(psi1.tensors[i][j])
            A2 = psi2.tensors[i][j]
            
            # Contract them, and combine indices
            A = contract(A1, A2, 4, 4)
            for idx in range(4):
                A, cmb = combineIdxs(A, [0, 4-idx])
            
            # Update the tensor
            psiCollapsed.tensors[i][j] = A
    #psiCollapsed.truncate(maxdim=chi)
    
    # Collapse all columns into a tensor
    psi = mps(1, psi1.length[1])
    psi.tensors = [np.ones((1, 1, 1))] * psi.length
    psi.center = 0
    rev = 0
    for i in range(psi1.length[0]):
        for j in range(psi1.length[1]):
            # Find the site
            site = j if not rev else psi.length - 1 - j
            
            # Contract as MPS-MPO
            prod = contract(psi.tensors[site], psiCollapsed.tensors[i][site], 1, 1)
            prod, cmb = combineIdxs(prod, [0, 2])
            prod, cmb = combineIdxs(prod, [0, 2])
            prod = permute(prod, 0, 1)
            
            # Update tensor
            psi.tensors[site] = prod
            
            # Truncate
            if j > 0:
                if not rev:
                    prod = contract(psi.tensors[site-1], prod, 2, 0)
                    psi.replacebond(site-1, prod, maxdim=chi, cutoff=cutoff)
                else:
                    prod = contract(prod, psi.tensors[site+1], 2, 0)
                    psi.replacebond(site, prod, rev, maxdim=chi, cutoff=cutoff)
            
        
        # Reverse direction
        rev = 1 - rev
    
    # Contract all the tensors in the MPS
    prod = np.ones((1, 1, 1))
    for i in range(psi.length):
        prod = contract(prod, psi.tensors[i], 2, 0)
        dims = (np.shape(prod)[0], 1, np.shape(psi.tensors)[2])
        prod = np.reshape(prod, dims)
    
    return np.asscalar(prod)


