# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from scipy.linalg import expm


def tensor(dims, data=None, dtype=np.complex128):
    # Set dimensions
    if isinstance(dims, int):
        dims = (dims, )

    # Create structure
    if data is None:
        return np.zeros(dims, dtype=dtype)
    else:
        return np.asarray(data, dtype=dtype)              


def randomTensor(dims, dtype=np.complex128):
    data = np.random.normal(size=dims).astype(dtype)
    if dtype == np.complex128:
        data += 1j*np.random.normal(size=dims).astype(dtype)
    return tensor(dims, data)


def contract(x, y, idx1, idx2):
    if np.shape(x)[idx1] == np.shape(y)[idx2]:
        storage = np.tensordot(x, y, axes=(idx1, idx2))
        return tensor(np.shape(storage), storage, storage.dtype)
    else:
        raise("The dimensions of contracting indexes do not match.")
        return 0


def trace(x, idx1, idx2):
    if np.shape(x)[idx1] == np.shape(x)[idx2]:
        storage = np.trace(x, axis1=idx1, axis2=idx2)
        return tensor(np.shape(storage), storage, storage.dtype)
    else:
        raise("The dimensions of contracting indexes do not match.")
        return 0


def permute(x, idx, position=-1):
    # Get the properties from the tensor
    dims = np.shape(x)
    storage = copy.deepcopy(x)
    
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


def combineIdxs(x, idxs):
    # Make a copy
    y = copy.deepcopy(x)
    
    # Get the dimension sizes from each idx
    dims = []
    for idx in idxs:
        dims.append(np.shape(x)[idx])
    
    # Permute indexs to the end
    i = 0
    for idx in idxs:
        y = permute(y, idx-i)
        i += 1
    
    # Get the current shape and then the new shape
    currentDims = np.shape(y)
    newDims = np.zeros(len(currentDims)-len(idxs)+1, dtype=int)
    newDims[0:len(currentDims)-len(idxs)] = currentDims[0:len(currentDims)-len(idxs)]
    newDims[-1] = np.prod(currentDims[len(currentDims)-len(idxs):])
    
    # Reshape
    storage = np.reshape(y, newDims)
    
    # Return tensor
    return tensor(tuple(newDims), storage, y.dtype), [idxs, dims]


def uncombineIdxs(x, comb):
    # make a copy
    y = copy.deepcopy(x)
    offset = len(np.shape(y))-1
    
    # Reshape dimensions back in
    newDims = np.zeros(offset + len(comb[1]), dtype=int)
    newDims[0:offset] = np.shape(y)[0:offset]
    newDims[offset:] = comb[1]
    storage = np.reshape(y, newDims)
    y = tensor(newDims, storage, y.dtype)
    
    # Permute indices back to the correct place
    idxs = comb[0]
    for i in range(len(idxs)):
        idxidx = np.argmin(idxs)
        idx = idxs[idxidx]
        y = permute(y, offset+idxidx+i, idx)
        idxs = np.delete(idxs, idxidx)
    
    return y


def svd(x, idx=-1, mindim=1, maxdim=0, cutoff=0):    
    # Make a copy
    y = copy.deepcopy(x)
    
    # Get the number of dims
    size = len(np.shape(y))
    
    # Check to see if index axis == -1
    if idx == -1:
        # Set to last
        idx = size - 1
    
    # Make sure axis is valid
    if not (isinstance(idx, int) and idx >= 0 and idx < size):
        print("axis must be an integer which refers to a valid index.")
        return 0
    
    # Group together all the indexs
    idxs = np.linspace(0, len(np.shape(y))-1, len(np.shape(y)), dtype=int)
    idxs = np.delete(idxs, idx)
    y, comb = combineIdxs(y, idxs)
    
    # Move decompose tensor to the end
    y = permute(y, 0)
    
    # Apply SVD
    U, S, V = np.linalg.svd(y)
    
    # Find how many singular values we should keep
    mindim = min(mindim, np.size(S))
    if maxdim == 0:
        maxdim = np.size(S)
    maxdim = min(np.size(S), maxdim)
    if cutoff != 0:
        S2 = S**2
        S2cum = np.flip(np.cumsum(np.flip(S2)) / np.sum(S2))
        idx = np.where(S2cum > cutoff)[0][-1]
        #if idx == 0:
        #    idx = np.size(S)
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
    x = tensor.flatten()
    return np.sqrt(np.dot(x, np.conj(x)))    


def ones(dims, dtype=np.complex128):
    return tensor(dims, np.ones(dims), dtype)


def reshape(x, shape):
    storage = np.reshape(x, shape)
    return tensor(shape, storage, x.dtype)


def exp(x, idxs, t = 1):
    if not isinstance(idxs, list):
        idxs = [idxs]
    
    x *= t
    x, C1 = combineIdxs(x, idxs)
    x, C2 = combineIdxs(x, [i for i in range(len(np.shape(x))-1)])
    x = permute(x, 1, 0)
    storage = expm(x)
    y = tensor(np.shape(x), storage, x.dtype)
    y = permute(y, 1, 0)
    y = uncombineIdxs(y, C2)
    y = uncombineIdxs(y, C1)
    return y
    

def dag(x):
    return np.conj(x)