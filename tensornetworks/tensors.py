""" 
    Provides functions to create tensors (as numpy arrays) and manipulate them.
"""

# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from scipy.linalg import expm


def tensor(dims, data=None, dtype=np.complex128):
    """
    Create a tensor.

    Parameters
    ----------
    dims : tuple
        Dimensions of the tensor
    data : multi-dimensional array (or np array), optional
        The information the tensor carries. The default is None.
    dtype : datatype, optional
        The datatype for the tensor.. The default is np.complex128.

    Returns
    -------
    np.ndarray
        Multi-dimensional np array.

    """
    
    # Set dimensions
    if isinstance(dims, int):
        dims = (dims, )

    # Create structure
    if data is None:
        return np.zeros(dims, dtype=dtype)
    else:
        return np.asarray(data, dtype=dtype)              


def randomTensor(dims, dtype=np.complex128):
    """
    Create a tensor with random entries

    Parameters
    ----------
    dims : tuple
        Dimensions of the tensor
    dtype : datatype, optional
        The datatype for the tensor.. The default is np.complex128.

    Returns
    -------
    np.ndarray
        Multi-dimensional np array.

    """
    
    # Draw an array of random numbers.
    data = np.random.normal(size=dims).astype(dtype)
    if dtype == np.complex128:
        data += 1j*np.random.normal(size=dims).astype(dtype)
    return tensor(dims, data)


def contract(x, y, idx1, idx2):
    """
    Contract two tensors.

    Parameters
    ----------
    x : np.ndarray
        First tensor.
    y : np.ndarray
        Second tensor.
    idx1 : int
        Index of first tensor.
    idx2 : int
        Index of second tensor.

    Returns
    -------
    np.ndarray
        Contracted tensors.

    """
    
    # Ensure the contracting dimensions are the same size.
    if np.shape(x)[idx1] == np.shape(y)[idx2]:
        storage = np.tensordot(x, y, axes=(idx1, idx2))
        return tensor(np.shape(storage), storage, storage.dtype)
    else:
        raise ValueError("The dimensions of contracting indexes do not match.")


def trace(x, idx1, idx2):
    """
    Take the trace of two dimensions in a tensor.

    Parameters
    ----------
    x : np.ndarray
        Tensor.
    idx1 : int
        First index.
    idx2 : int
        Second index.

    Returns
    -------
    np.ndarray
        Tensor with the dimensions traced.

    """
    
    # Ensure the dimensions are the same size
    if np.shape(x)[idx1] == np.shape(x)[idx2]:
        storage = np.trace(x, axis1=idx1, axis2=idx2)
        return tensor(np.shape(storage), storage, storage.dtype)
    else:
        raise ValueError("The dimensions of contracting indexes do not match.")
        return 0


def permute(x, idx, position=-1):
    """
    Permute an index in a tensor to a different position.

    Parameters
    ----------
    x : np.ndarray
        Tensor.
    idx : int
        Index to permute.
    position : int, optional
        The position to permute. The default is -1, which permutes to the end.

    Returns
    -------
    np.ndarray
        Tensor with permuted index.

    """
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
    """
    Combine the indexs in a tensor.

    Parameters
    ----------
    x : np.ndarray
        Tensor.
    idxs : list of ints
        List of indexs to group.

    Returns
    -------
    np.ndarray
        Tensor with indexs grouped together.
    list
        Contains a list of of the grouped indexs and their dimensions, used
        to uncombine them.

    """
    
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
    """
    Uncombine the indexs of a tensor.

    Parameters
    ----------
    x : np.ndarray
        Tensor.
    comb : list
        Uncombiner list from combineIdx.

    Returns
    -------
    y : np.ndarray
        Tensor with indexs uncombined.

    """
    
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
    """
    Singular value decomposition of a tensor.

    Parameters
    ----------
    x : np.ndarray
        Tensor to apply SVD to.
    idx : int, optional
        Dimension to apply SVD to. The default is -1 (last index).
    mindim : int, optional
        The minimum number of singular values to keep. The default is 1.
    maxdim : int, optional
        The maximum number of singular values to keep. The default is 0, which
        defines no upper limit.
    cutoff : float, optional
        The truncation error of singular values. The default is 0.

    Returns
    -------
    U, S, V
        Tensors from singular value decomposition.

    """
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
        raise ValueError("axis must be an integer which refers to a valid index.")
    
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
    if cutoff != 0 and np.sum(S) > 10**-16:
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
    """
    Calulate the norm of a tensor by contracting its indices with itself.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor

    Returns
    -------
    np.complex64
        The calculated norm of the tensor

    """
    x = tensor.flatten()
    return np.sqrt(np.dot(x, np.conj(x)))    


def ones(dims, dtype=np.complex128):
    """
    Create a tensor filled with ones.

    Parameters
    ----------
    dims : tuple
        Dimensions of the tensor.
    dtype : type, optional
        The data type of the tensor. The default is np.complex128.

    Returns
    -------
    np.ndarray
        The tensor filled with ones.

    """
    return tensor(dims, np.ones(dims), dtype)


def shape(x):
    """
    Find the dimensions of a tensor.

    Parameters
    ----------
    x : np.ndarray
        The tensor.

    Returns
    -------
    tuple
        Dimensions of the tensor.

    """
    return np.shape(x)


def reshape(x, shape):
    """
    Reshape a tensor.

    Parameters
    ----------
    x : np.ndarray
        The tensor.
    shape : tuple
        New dimensions of the tensor.

    Returns
    -------
    np.ndarray
        Reshaped tensor.

    """
    storage = np.reshape(x, shape)
    return tensor(shape, storage, x.dtype)


def exp(x, idxs, t = 1):
    """
    Calculate the exponential of a tensor by recasting into a square rank-2
    tensor, expontiating and moving back.

    Parameters
    ----------
    x : np.ndarray
        The tensor.
    idxs : list
        List of indexs to form the second dimension of the tensor.
    t : number, optional
        The time to multiply by in the exponential. The default is 1.

    Returns
    -------
    y : np.ndarray
        The exponentiated tensor.

    """
    
    # Check to see if the idxs are a list
    if not isinstance(idxs, list):
        idxs = [idxs]
    
    # Copy the tensor and multiply by time.
    x = copy.deepcopy(x)
    x *= t
    
    # Combine indexs and put into the correct shape
    x, C1 = combineIdxs(x, idxs)
    x, C2 = combineIdxs(x, [i for i in range(len(np.shape(x))-1)])
    x = permute(x, 1, 0)
    
    # Exponentiate
    y = expm(x)
    
    # Return to original shape
    y = permute(y, 1, 0)
    y = uncombineIdxs(y, C2)
    y = uncombineIdxs(y, C1)
    return y
    

def dag(x):
    """
    Calculate the complex conjugate of a tensor.

    Parameters
    ----------
    x : np.ndarray
        The tensor.

    Returns
    -------
    np.ndarray
        The complex conjugated tensor.

    """
    return np.conj(x)