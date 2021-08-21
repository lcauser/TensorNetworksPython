from tensornetworks.sitetypes import sitetypes
from tensornetworks.tensors import tensor, permute, shape, reshape
from tensornetworks.lattices.spinHalf import spinHalf
from tensornetworks.structures.mps import mps
from tensornetworks.structures.mpo import mpo
import numpy as np

def qKCMS(omega, gamma, kappa=1.0):
    """
    Creates a spin half with the states and projectors for 'light' and 'dark'
    states on qKCMs. 

    Parameters
    ----------
    omega : float
        Quantum noise.
    gamma : float
        Dissipation to light state.
    kappa : float, optional
        Dissipation to dark state. The default is 1.

    Returns
    -------
    sh : sitetypes
        Quantum KCMs site type.

    """
    # Create spin half class
    sh = spinHalf()
    
    # Find the light and dark states
    if gamma == kappa:
        light = np.asarray([1, 0])
        dark = np.asarray([0, 1])
    else:
        ss = (8*omega**2+(kappa+gamma)**2)**-1 * \
              np.array([[4*omega**2 + gamma*(kappa+gamma), -2j*omega*(kappa-gamma)],
                        [2j*omega*(kappa-gamma), 4*omega**2 + kappa*(kappa+gamma)]])
        eigs, vecs = np.linalg.eig(ss)
        lightEig = eigs[np.argmin(eigs)]
        light = vecs[:, np.argmin(eigs)]
        darkEig = eigs[np.argmax(eigs)]
        dark = vecs[:, np.argmax(eigs)]
    
    # Add the states and projectors to the state space.
    sh.addState("l", tensor(2, light))
    sh.addState("da", tensor(2, dark))
    sh.addOp("pl", tensor((2, 2), np.outer(light, np.conj(light))), "pl")
    sh.addOp("pda", tensor((2, 2), np.outer(dark, np.conj(dark))), "pda")
    
    return sh


def qKCMSDM(omega, gamma, kappa=1):
    """ Density matrix representation for spin-half quantum Kinetically 
    Constrained Models.
    
    Parameters
    ----------
    omega : float
        Quantum noise.
    gamma : float
        Dissipation to light state.
    kappa : float, optional
        Dissipation to dark state. The default is 1.
        
    Returns
    ----------
    st : sitetypes1d
        Site types.
    """
    # Load spin half and fetch the light and dark states
    sh = qKCMS(omega, gamma, kappa)
    light = sh.state("l")
    dark = sh.state("da")
    
    # Make sitetype
    st = sitetypes(4)
    
    # Add states
    st.addState("up", np.kron(sh.state("up"), sh.state("up")))
    st.addState("dn", np.kron(sh.state("dn"), sh.state("dn")))
    st.addState("l", np.kron(sh.state("l"), sh.state("l")))
    st.addState("da", np.kron(sh.state("da"), sh.state("da")))
    
    # Add sigma operators
    st.addOp("xid", np.kron(sh.op("x"), sh.op("id")), "xid")
    st.addOp("idx", np.kron(sh.op("id"), sh.op("x")), "idx")
    st.addOp("yid", np.kron(sh.op("y"), sh.op("id")), "yid")
    st.addOp("idy", np.kron(sh.op("id"), np.transpose(sh.op("y"))), "idx")
    st.addOp("zid", np.kron(sh.op("z"), sh.op("id")), "zid")
    st.addOp("idz", np.kron(sh.op("id"), sh.op("z")), "idz")
    st.addOp("idid", np.kron(sh.op("id"), sh.op("id")), "idid")
    
    # Add spin projectors
    st.addOp("puid", np.kron(sh.op("pu"), sh.op("id")), "puid")
    st.addOp("idpu", np.kron(sh.op("id"), np.transpose(sh.op("pu"))), "puid")
    st.addOp("pdid", np.kron(sh.op("pd"), sh.op("id")), "pdid")
    st.addOp("idpd", np.kron(sh.op("id"), np.transpose(sh.op("pd"))), "pdid")
    
    # Add jump
    st.addOp("s+id", np.kron(sh.op("s+"), sh.op("id")), "s-id")
    st.addOp("ids+", np.kron(sh.op("id"), sh.op("s+")), "ids-")
    st.addOp("s-id", np.kron(sh.op("s-"), sh.op("id")), "s+id")
    st.addOp("ids-", np.kron(sh.op("id"), sh.op("s-")), "ids+")
    st.addOp("s-s-", np.kron(sh.op("s-"), sh.op("s-")), "s+s+")
    st.addOp("s+s+", np.kron(sh.op("s+"), sh.op("s+")), "s+s+")
    
    # Add light and dark operators
    st.addOp("plid", np.kron(sh.op("pl"), sh.op("id")), "plid")
    st.addOp("idpl", np.kron(sh.op("id"), np.transpose(sh.op("pl"))),
             "plid")
    st.addOp("plpl", np.kron(sh.op("pl"), np.transpose(sh.op("pl"))),
             "plpl")
    
    
    return st


def vectoDM(psi : mps):
    """
    Turns an MPS into density matrix form

    Parameters
    ----------
    psi : mps

    Returns
    -------
    rho : mpo

    """
    
    # Create the MPO
    rho = mpo(2, psi.length)
    
    # Loop through each tensor, and split the tensor into the physical dims
    for i in range(psi.length):
        A = psi.tensors[i]
        O = permute(A, 1)
        O = reshape(O, (shape(A)[0], shape(A)[2], 2, 2))
        O = permute(O, 1)
        rho.tensors[i] = O
    
    return rho
        