# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from scipy.linalg import expm

class tensor:
    
    def __init__(self, dims, data=None, dtype=None):
        
        # Set dimensions
        if isinstance(dims, int):
            dims = (dims, )
        self.dims = dims
        
        # Set data type
        if dtype == None:
            if data is None:
                self.dtype = np.complex128
            else:
                self.dtype = data.dtype
        else:
            self.dtype = dtype
        
        # Create structure
        self.createStructure()
        if data is not None:
            if np.all(self.dims == np.shape(data)):
                self.storage = data
            
        return None
    
    
    def __add__(self, x):
        # Check to see if dimensions are the same
        if x.dims != self.dims:
            raise Exception("Tensors have different dimensions")
            return False
        
        # Create new tensor
        if x.dtype == np.float64 or self.dtype == np.float64:
            dtype = np.float64
        else:
            dtype = np.complex128
        
        z = tensor(x.dims, dtype=dtype)
        z.storage = x.storage + self.storage
        
        return z    
        
    def __sub__(self, x):
        # Check to see if dimensions are the same
        if x.dims != self.dims:
            raise Exception("Tensors have different dimensions")
            return False
        
        # Create new tensor
        if x.dtype == np.complex64 or self.dype == np.complex64:
            dtype = np.complex64
        else:
            dtype = np.float64
        
        z = tensor(x.dims, dtype=dtype)
        z.storage = x.storage - self.storage
        return z
    
    
    def __mul__(self, other):
        if isinstance(other, (float, complex, int, np.complex, np.float, np.int)):
            x = copy.deepcopy(self)
            x.storage *= other
            return x
    
    def __div__(self, other):
        if isinstance(other, (float, complex, int, np.complex, np.float, np.int)):
            x = copy.deepcopy(self)
            x.storage /= other
            return x
        
    def __rmul__(self, other):
        if isinstance(other, (float, complex, int, np.complex, np.float, np.int)):
            x = copy.deepcopy(self)
            x.storage *= other
            return x
    
        
    def createStructure(self):
        self.storage = np.zeros(self.dims, dtype=self.dtype)
        
    


def randomTensor(dims, dtype=np.complex64):
    data = np.random.normal(size=dims).astype(dtype)
    if dtype == np.complex64:
        data += 1j*np.random.normal(size=dims).astype(dtype)
    return tensor(dims, data)


def dag(x : tensor):
    y = copy.deepcopy(x)
    y.storage = np.conj(y.storage)
    return y


def contract(x : tensor, y : tensor, idx1, idx2):
    if x.dims[idx1] == y.dims[idx2]:
        storage = np.tensordot(x.storage, y.storage, axes=(idx1, idx2))
        return tensor(np.shape(storage), storage, storage.dtype)
    else:
        raise("The dimensions of contracting indexes do not match.")
        return 0

def trace(x : tensor, idx1, idx2):
    if x.dims[idx1] == x.dims[idx2]:
        storage = np.trace(x.storage, axis1=idx1, axis2=idx2)
        return tensor(np.shape(storage), storage, storage.dtype)
    else:
        raise("The dimensions of contracting indexes do not match.")
        return 0

def permute(x : tensor, idx, position=-1):
    # Get the properties from the tensor
    dims = copy.deepcopy(x.dims)
    storage = copy.deepcopy(x.storage)
    
    # Check to see if position is -1, if so get the end axis position
    if position == -1:
        position = len(dims) - 1
        
    # Loop through from it's current position to the desired position, swapping
    # axes
    for i in range(np.abs(position - idx)):
        if position > idx:
            storage = np.swapaxes(storage, axis1 = idx+i, axis2 = idx+i+1)
        elif position < idx:
            storage = np.swapaxes(storage, axis1 = idx-i, axis2 = idx-i-1)
    
    return tensor(np.shape(storage), storage, x.dtype)


def combineIdxs(x : tensor, idxs):
    # Make a copy
    y = copy.deepcopy(x)
    
    # Get the dimension sizes from each idx
    dims = []
    for idx in idxs:
        dims.append(x.dims[idx])
    
    # Permute indexs to the end
    i = 0
    for idx in idxs:
        y = permute(y, idx-i)
        i += 1
    
    # Get the current shape and then the new shape
    currentDims = y.dims
    newDims = np.zeros(len(currentDims)-len(idxs)+1, dtype=int)
    newDims[0:len(currentDims)-len(idxs)] = currentDims[0:len(currentDims)-len(idxs)]
    newDims[-1] = np.prod(currentDims[len(currentDims)-len(idxs):])
    
    # Reshape
    storage = np.reshape(y.storage, newDims)
    
    # Return tensor
    return tensor(tuple(newDims), storage, y.dtype), [idxs, dims]


def uncombineIdxs(x : tensor, comb):
    # make a copy
    y = copy.deepcopy(x)
    offset = len(y.dims)-1
    
    # Reshape dimensions back in
    newDims = np.zeros(offset + len(comb[1]), dtype=int)
    newDims[0:offset] = y.dims[0:offset]
    newDims[offset:] = comb[1]
    storage = np.reshape(y.storage, newDims)
    y = tensor(newDims, storage, y.dtype)
    
    # Permute indices back to the correct place
    idxs = comb[0]
    for i in range(len(idxs)):
        idxidx = np.argmin(idxs)
        idx = idxs[idxidx]
        y = permute(y, offset+idxidx+i, idx)
        idxs = np.delete(idxs, idxidx)
    
    return y


def svd(x : tensor, idx=-1, mindim=1, maxdim=0, cutoff=0):
    """ Apply a singular value decomposition to a tensor.
    Args: 
        tensor = tensor of any rank
        idx = which index you want to decompose, default = -1 (last)
    Returns:
        U = tensor with untouched indices and decomposed index
        S = matrix of singular values
        V = matrix of decomposed index
    """
    
    # Make a copy
    y = copy.deepcopy(x)
    
    # Get the number of dims
    size = len(y.dims)
    
    # Check to see if index axis == -1
    if idx == -1:
        # Set to last
        idx = size - 1
    
    # Make sure axis is valid
    if not (isinstance(idx, int) and idx >= 0 and idx < size):
        print("axis must be an integer which refers to a valid index.")
        return 0
    
    # Group together all the indexs
    idxs = np.linspace(0, len(y.dims)-1, len(y.dims), dtype=int)
    idxs = np.delete(idxs, idx)
    y, comb = combineIdxs(y, idxs)
    
    # Move decompose tensor to the end
    y = permute(y, 0)
    
    # Apply SVD
    U, S, V = np.linalg.svd(y.storage)
    
    # Find how many singular values we should keep
    mindim = min(mindim, np.size(S))
    if maxdim == 0:
        maxdim = np.size(S)
    maxdim = min(np.size(S), maxdim)
    if cutoff != 0:
        S2 = S**2
        S2cum = 1 - (np.cumsum(S2) / np.sum(S2))
        idx = np.argmax(S2cum < cutoff)
        if idx == 0:
            idx = np.size(S)
        maxdim = min(maxdim, idx+1)
    vals = max(maxdim, mindim)
    
    # Truncate
    U = U[:, 0:vals]
    S = np.diag(S[0:vals])
    V = V[0:vals, :]
    
    # Return to tensors
    U = tensor(np.shape(U), U, x.dtype)
    S = tensor(np.shape(S), S, x.dtype)
    V = tensor(np.shape(V), V, x.dtype)
    
    # Ungroup indexs
    U = permute(U, 1, 0)
    U = uncombineIdxs(U, comb)
    
    return U, S, V
        
    
def norm(tensor):
    x = tensor.storage.flatten()
    return np.sqrt(np.dot(x, np.conj(x)))    


def ones(dims, dtype=np.complex128):
    return tensor(dims, np.ones(dims), dtype)


def reshape(x, shape):
    storage = np.reshape(x.storage, shape)
    return tensor(shape, storage, x.dtype)


def exp(x, idxs, t = 1):
    if not isinstance(idxs, list):
        idxs = [idxs]
    
    x *= t
    x, C1 = combineIdxs(x, idxs)
    x, C2 = combineIdxs(x, [i for i in range(len(x.dims)-1)])
    x = permute(x, 1, 0)
    storage = expm(x.storage)
    y = tensor(x.dims, storage, x.dtype)
    y = permute(y, 1, 0)
    y = uncombineIdxs(y, C2)
    y = uncombineIdxs(y, C1)
    return y
    
    