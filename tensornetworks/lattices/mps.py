from tensornetworks.sitetypes1d import sitetypes1d
from tensornetworks.tensors import tensor
import numpy as np

def spinHalf():
    """ S = 1/2 typeset
    
    Returns:
        st (sitetypes1d): Spin half typeset.
    """
    
    # Make sitetype
    st = sitetypes1d(2)
    
    # Add states
    st.addState("up", tensor(2, np.asarray([1, 0], dtype=np.complex128)))
    st.addState("dn", tensor(2, np.asarray([0, 1], dtype=np.complex128)))
    st.addState("s", tensor(2, np.sqrt(0.5)*np.asarray([1, 1], dtype=np.complex128)))
    st.addState("as", tensor(2, np.sqrt(0.5)*np.asarray([1, -1], dtype=np.complex128)))
    
    # Add operators
    st.addOp("x", tensor((2, 2), np.array([[0, 1], [1, 0]], dtype=np.complex128)), "x")
    st.addOp("y", tensor((2, 2), np.array([[0, -1j], [1j, 0]], dtype=np.complex128)), "y")
    st.addOp("z", tensor((2, 2), np.array([[1, 0], [0, -1]], dtype=np.complex128)), "z")
    st.addOp("id", tensor((2, 2), np.array([[1, 0], [0, 1]], dtype=np.complex128)), "id")
    st.addOp("n", tensor((2, 2), np.array([[1, 0], [0, 0]], dtype=np.complex128)), "n")
    st.addOp("pu", tensor((2, 2), np.array([[1, 0], [0, 0]], dtype=np.complex128)), "pu")
    st.addOp("pd", tensor((2, 2), np.array([[0, 0], [0, 1]], dtype=np.complex128)), "pd")
    st.addOp("s+", tensor((2, 2), np.array([[0, 1], [0, 0]], dtype=np.complex128)), "s-")
    st.addOp("s-", tensor((2, 2), np.array([[0, 0], [1, 0]], dtype=np.complex128)), "s+")
    
    return st