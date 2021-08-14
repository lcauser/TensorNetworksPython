"""
    The class for matrix product states (mps), and functions to manipulate
    and work with them.
"""

# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.sitetypes1d import sitetypes1d

class mps:
    
    def __init__(self, dim, length, dtype=np.complex128):
        """
        Create a matrix product state class.

        Parameters
        ----------
        dim : int
            Physical dimension of each site on the MPS.
        length : int
            Length of the MPS.
        dtype : type, optional
            Type of the tensors within the MPS. The default is np.complex128.

        """
        
        # Store the relevent information for the MPS
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
        """ Creates the structure of an MPS, full of zero tensors. """
        
        tensors = []
        for i in range(self.length):
            tensors.append(tensor((1, self.dim, 1)))
        self.tensors = tensors
        return None
    
    
    def bondDim(self, idx):
        """
        Find the bond dimension after site idx

        Parameters
        ----------
        idx : int
            Site number.

        Returns
        -------
        int
            Bond dimension.

        """
        return shape(self.tensors[idx])[2]
    
    
    def maxBondDim(self):
        """
        Find the maximum bond dimension within the MPS.

        Returns
        -------
        dim : int
            Maximum bond dimension.

        """
        dim = 1
        for idx in range(self.length):
            dim = max(dim, self.bondDim(idx))
        return dim
            
    
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
        A = self.tensors[idx-1]
        A = contract(A, V, 2, 1)
        A = contract(A, S, 2, 1)
        
        self.tensors[idx-1] = A
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
        U, S, V = svd(self.tensors[idx], 2, mindim, maxdim, cutoff)
        
        # Get the site to the right
        A = self.tensors[idx+1]
        S = contract(S, V, 1, 0)
        A = contract(S, A, 1, 0)
        
        self.tensors[idx+1] = A
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
    
    
    def norm(self):
        """
        Efficiently calculate the norm of the MPS through the orthogonal
        center.

        Returns
        -------
        norm: float/complex
            Norm of the MPS

        """
        # Orthogonalize
        if self.center == None:
            self.orthogonalize(0)
        
        return norm(self.tensors[self.center])
    
    
    def normalize(self):
        """ Normalize the MPS. """
        
        # Calculate the norm and rescale the orthogonal centre.
        norm = self.norm()
        self.tensors[self.center] *= (1/norm)
        return None
    
    
    def truncate(self, mindim=1, maxdim=0, cutoff=0):
        """
        Truncate the MPS though SVDs 

        Parameters
        ----------
        mindim : int, optional
            The minimum number of singular values to keep. The default is 1.
        maxdim : int, optional
            The maximum number of singular values to keep. The default is 0,
            which defines no upper limit.
        cutoff : float, optional
            The truncation error of singular values. The default is 0.

        """
        # If not at the edge, move orthogonal centre to the edge
        if self.center != 0 and self.center != self.length-1:
            self.orthogonalize(self.length-1)
        
        # Move orthogonal centre across the entire MPS, applying SVD truncation
        self.orthogonalize(self.length-1 - self.center, mindim, maxdim, cutoff)
        return self
    
    
    def replacebond(self, site, A, rev=False, mindim=1, maxdim=0, cutoff=0,
                    nsites = 2):
        """
        Replace the bond in the MPS with contracted tensors via SVD.

        Parameters
        ----------
        site : int
            First site in the tensor.
        A : np.ndarray
            tensor
        rev : bool, optional
            False = sweeping right, true = sweeping left. The default is False.
        mindim : int, optional
            The minimum number of singular values to keep. The default is 1.
        maxdim : int, optional
            The maximum number of singular values to keep. The default is 0,
            which defines no upper limit.
        cutoff : float, optional
            The truncation error of singular values. The default is 0.
        nsites : int, optional
            The number of sites in the tensor. The default is 2.

        """
        # Deal with nsites = 1
        if nsites == 1:
            # Update the tensor
            self.tensors[site] = A
            
            # Move center
            if rev and site != 0:
                self.orthogonalize(site-1)
            if (not rev) and site != self.length-1:
                self.orthogonalize(site+1)
            return self
        
        # Loop through applying SVDs
        for i in range(nsites-1):
            if rev:
                # Deal with sweeping left
                # Group together last indexs
                A, C1 = combineIdxs(A, [len(shape(A))-2, len(shape(A))-1])
                
                # Find the site to update
                site1 = site+nsites-1-i
                
                # Apply SVD
                U, S, V = svd(A, -1, mindim, maxdim, cutoff)
                
                # Restore U
                U = contract(U, S, len(shape(U))-1, 0)
                
                # Restore V and update tensor
                D = 1 if site1 == self.length - 1 else shape(self.tensors[site1+1])[0]
                V = reshape(V, (shape(S)[1], self.dim, D))
                self.tensors[site1] = V
            else:
                # Deal with sweeping right
                # Group first two indexs toegether
                A, C1 = combineIdxs(A, [0, 1])
                
                # Find the site to update
                site1 = site + i
                
                # Apply SVD
                U, S, V = svd(A, -1, mindim, maxdim, cutoff)
                
                # Restore U
                U = contract(U, S, len(shape(U))-1, 0)
                U = permute(U, len(shape(U))-1, 0)
                
                # Restore V and update tensor
                D = 1 if site1 == 0 else shape(self.tensors[site1-1])[2]
                V = reshape(V, (np.shape(S)[1], D, self.dim))
                V = permute(V, 0)
                self.tensors[site1] = V
                
            # Update A
            A = U
        
        # Find the final site and update the tensor and center
        site1 = site if rev else site + nsites - 1
        self.tensors[site1] = U
        self.center = site1
        
        return self
        


def randomMPS(dim, length, bondDim=1, normalize=True, dtype=np.complex128):
    """
    Create a MPS with random tensors.

    Parameters
    ----------
    dim : int
        Physical dimension of each site on the MPS.
    length : int
        Length of the MPS.
    bondDim : int, optional
        Bond dimension of the MPS. The default is 1.
    normalize : bool, optional
        Should the MPS be normalized. The default is True.
    dtype : type, optional
        Type of the tensors within the MPS. The default is np.complex128.

    Returns
    -------
    psi : mps
        Random MPS.

    """
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
    psi.tensors[0] = randomTensor(shape(psi.tensors[0]), dtype)
    if normalize:
        psi *= 1 / psi.norm()    
    return psi


def meanfieldMPS(dim, length, A, dtype=np.complex128):
    """
    Create an MPS with identical tensors (bond dimension one).

    Parameters
    ----------
    dim : int
        Physical dimension of each site on the MPS.
    length : int
        Length of the MPS.
    A : np.ndarray
        Rank one tensor of size dim
    dtype : type, optional
        Type of the tensors within the MPS. The default is np.complex128.

    Returns
    -------
    psi : MPS
        Mean field MPS.

    """
    # Deal with A
    if len(shape(A)) == 1:
        if shape(A)[0] == dim:
            A = np.reshape(A, (1, np.size(A), 1))
        else:
            raise ValueError("The tensor does not have the correct dimensions.")
    elif not shape(A) == (1, dim, 1):
        raise ValueError("The tensor does not have the correct dimensions.")
    A = tensor(shape(A), A, dtype)
    
    # Create MPS
    psi = mps(dim, length, dtype)
    for i in range(length):
        psi.tensors[i] = copy.deepcopy(A)
    
    return psi


def productMPS(s : sitetypes1d, states):
    """
    Create a product (bond dimension one) MPS from a list of states.

    Parameters
    ----------
    s : sitetypes1d
        Site types with state names.
    states : list
        List of states.

    Returns
    -------
    psi : mps
        Product MPS.

    """
    # Check list of states
    if not isinstance(states, list):
        raise ValueError("States must be a list of product states.")
    
    # Create mps
    psi = mps(s.dim, len(states))
    for i in range(len(states)):
        psi.tensors[i][0, :, 0] = s.state(states[i])
    
    return psi


def dot(psi : mps, phi : mps):
    """
    Calculate the inner product between two MPS.

    Parameters
    ----------
    psi : mps
    phi : mps

    Returns
    -------
    product: number
        Scalar product of two MPS.

    """
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
    