"""
    Class to create matrix product operators (MPOs), manipulate them and use
    them with matrix product states.
"""

# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.mps import mps

class mpo:
    
    def __init__(self, dim, length, dtype=np.complex128):
        """
        Create a matrix product operator.

        Parameters
        ----------
        dim : int
            Physical dimension of each site on the MPO.
        length : int
            Length of the MPO.
        dtype : type, optional
            Type of the tensors within the MPO. The default is np.complex128.

        """
        # Store relevent information
        self.dim = dim
        self.length = length
        self.dtype = dtype
        self.center = None
        
        # Create the structure of the mps
        self.createStructure()
    
    
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
        """ Creates the structure of the MPO, full of zero tensors. """
        tensors = []
        for i in range(self.length):
            tensors.append(tensor((1, self.dim, self.dim, 1)))
        self.tensors = tensors
        return self
    
    
    def moveLeft(self, idx, mindim=1, maxdim=0, cutoff=0):
        """
        Move the gauge from site idx to the left.

        Parameters
        ----------
        idx : int
            Site index
        mindim : int, optional
            The minimum number of singular values to keep. The default is 1.
        maxdim : int, optional
            The maximum number of singular values to keep. The default is 0,
            which defines no upper limit.
        cutoff : float, optional
            The truncation error of singular values. The default is 0.

        """
        # Get the SVD at the site
        U, S, V = svd(self.tensors[idx], 0, mindim, maxdim, cutoff)
        
        # Get the site to the left
        M = self.tensors[idx-1]
        M = contract(M, V, 3, 1)
        M = contract(M, S, 3, 1)
        
        self.tensors[idx-1] = M
        self.tensors[idx] = U
        
        return self
    
    def moveRight(self, idx, mindim=1, maxdim=0, cutoff=0):
        """
        Move the gauge from site idx to the right.

        Parameters
        ----------
        idx : int
            Site index
        mindim : int, optional
            The minimum number of singular values to keep. The default is 1.
        maxdim : int, optional
            The maximum number of singular values to keep. The default is 0,
            which defines no upper limit.
        cutoff : float, optional
            The truncation error of singular values. The default is 0.

        """
        # Get the SVD at the site
        U, S, V = svd(self.tensors[idx], 3, mindim, maxdim, cutoff)
        
        # Get the site to the right
        M = self.tensors[idx+1]
        S = contract(S, V, 1, 0)
        M = contract(S, M, 1, 0)
        
        self.tensors[idx+1] = M
        self.tensors[idx] = U
        
        return self
    
    
    def orthogonalize(self, idx, mindim=1, maxdim=0, cutoff=0):
        """
        Move the orthogonal centre to idx.

        Parameters
        ----------
        idx : int
            Site index
        mindim : int, optional
            The minimum number of singular values to keep. The default is 1.
        maxdim : int, optional
            The maximum number of singular values to keep. The default is 0,
            which defines no upper limit.
        cutoff : float, optional
            The truncation error of singular values. The default is 0.

        """
        
        # Check to see if already in a canonical form
        if self.center == None:
            # Move from both the left and right to the correct site
            for i in range(idx-1):
                self.moveRight(i, mindim, maxdim, cutoff)
            
            for i in range(self.length-1-idx):
                self.moveLeft(self.length-1-i, mindim, maxdim, cutoff)
        else:
            if idx < self.center:
                # Move left to the appropiate site
                for i in range(self.center-idx):
                    self.moveLeft(self.center-i, mindim, maxdim, cutoff)
            
            if idx > self.center:
                # Move right to the appropiate site
                for i in range(idx-self.center):
                    self.moveRight(self.center+i, mindim, maxdim, cutoff)
                    
        self.center = idx
        return self
    
    
    def bondDim(self, idx):
        """
        Find the bond dimension after site idx.

        Parameters
        ----------
        idx : int
            Site number.

        Returns
        -------
        int
            Bond dimension.

        """
        return shape(self.tensors[idx])[3]
    
    
    def maxBondDim(self):
        """
        Find the maximum bond dimension within the MPO.

        Returns
        -------
        dim : int
            Maximum bond dimension.

        """
        dim = 1
        for idx in range(self.length):
            dim = max(dim, self.bondDim(idx))
        return dim
        
    
def uniformMPO(dim, length, M, dtype=np.float64):
    """
    Create a uniform MPO from tensor M.

    Parameters
    ----------
    dim : int
        Physical dimension of each site on the MPO.
    length : int
        Length of the MPO.
    M : np.ndarray
        Rank-4 tensor, with dimensions 1 and 2 being size dim.
    dtype : type, optional
        Type of the tensors within the MPO. The default is np.complex128.

    Returns
    -------
    O : mpo
        The uniform MPO.

    """
    # Check M is of the correct form
    if len(shape(M)) != 4:
        raise ValueError("M must be a rank-4 tensor.")
    if shape(M)[0] != shape(M)[3]:
        raise ValueError("Tensor must have equal bond dimensions.")
    if shape(M)[1] != shape(M)[2] or shape(M)[1] != dim:
        raise ValueError("Tensor must have the same physical dimensions.")        
    
    # Crate the MPO
    bonddim = shape(M)[0]
    O = mpo(dim, length, dtype)
    O.tensors[0] = M[bonddim-1:bonddim, :, :, :]
    for i in range(length-2):
        O.tensors[1+i] = M
    O.tensors[length-1] = M[:, :, :, 0:1]
    return O


def bMPO(length):
    """
    Create a boundary MPO of ones for PEPS environments.

    Parameters
    ----------
    length : int
        Length of MPO.

    Returns
    -------
    bMPO : mpo
        boundary MPO.

    """
    bMPO = mpo(1, length)
    for i in range(length):
        bMPO.tensors[i] = np.ones((1, 1, 1, 1))
    return bMPO


def applyMPO(O : mpo, psi : mps):
    """
    Apply a MPO to a MPS.

    Parameters
    ----------
    O : mpo
    psi : mps

    Returns
    -------
    psi2 : mps
        Modifed MPS.

    """
    psi2 = copy.deepcopy(psi)
    for i in range(O.length):
        A = contract(O.tensors[i], psi.tensors[i], 2, 1)
        A, c = combineIdxs(A, [0, 3])
        A, c = combineIdxs(A, [1, 2])
        A = permute(A, 0, 1)
        psi2.tensors[i] = A
    
    return psi2


def inner(psi1 : mps, O : mpo, psi2 : mps):
    """
    Calculate the inner product of an MPS, MPO and MPS.

    Parameters
    ----------
    psi1 : mps
    O : mpo
    psi2 : mps

    Returns
    -------
    number
        Inner product

    """
    # Make initial product tensor
    prod = ones((1, 1, 1))
    
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
    
    
        