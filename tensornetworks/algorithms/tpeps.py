"""
    Evolve a PEPS in time under some trotter gates, and use variational
    minimization to truncate, keeping bond dimensions small.
"""

import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.environment import environment
from scipy.sparse.linalg import LinearOperator, cg


def tpeps(psi, gate, dt, maxdim, maxiter=1000, chi=0, cutoff=10**-16, tol=10**-8):
    # Determine chi
    if chi == 0:
        chi = 10*maxdim**2
    
    # Create the environment and evolution gate
    env = environment(psi, chi)
    gate1 = exp(gate, [1, 3], dt)
    gate2 = exp(gate, [1, 3], dt/2)
    gate3 = exp(gate, [1, 3], dt/4)
    
    # Evolve until convergence
    converge = False
    k = 0
    check = 0 
    Zs = []
    while not converge:
        Z = 0
        maxChi = 0
        error = 0
        
        # Loop through horizontal, and then vertical bonds
        for d in [0, 1, 0]:
            # Loop through rows / columns
            for i in range(psi.length[d]):
                # Loop through the even / odd sites
                evens = []
                odds = []
                for j in range(psi.length[1-d]-1):
                    if j % 2 == 0:
                        evens.append(j)
                    else:
                        odds.append(j)
                
                for d2 in [0, 1, 0]:
                    # Determine the sites
                    sites = evens if d2 == 0 else np.flip(odds)
                    
                    # Determine the gates
                    if d == 0 and d2 == 0:
                        gates = gate3
                    elif d == 0 and d2 == 1:
                        gates = gate2
                    elif d == 1 and d2 == 0:
                        gates = gate2
                    elif d == 1 and d2 == 1:
                        gates = gate1
                    
                    # Loop through all sites
                    for site in sites:
                        site1 = i if d == 0 else site
                        site2 = site if d == 0 else i
                        
                        # Optimize
                        norm, err = optimize(env, site1, site2, d, d2, gates,
                                             maxdim, cutoff=cutoff)
                        Z += np.log(np.real(norm)) / (2*dt)
                        maxChi = max(maxChi, env.maxBondDim())
                        error = max(error, err)
        
        # Increment simulations and check convergence
        k += 1
        Zs.append(Z)
        if k >= 2:
            if np.abs(Zs[-1] - Zs[-2]) < tol:
                check += 1
            else:
                check = 0
        if check >= 1 or k >= maxiter:
            converge = True
        
        # Rescale the peps
        psi.rescale()
        
        # Output information
        print("dt="+str(dt)+", sim="+str(k)+", energy="+str(Z) + \
               ", maxbondim="+str(psi.maxBondDim())+", maxchi="+str(maxChi)+\
                   ", cost="+str(error))
    return psi, Zs[-1]
                    



def optimize(env, i, j, direction, direction2, gate, maxdim, cutoff=10**-16,
             tol=10**-5, maxiter=1000):
    # Detmine which sites to fetch
    site1 = [i, j]
    site2 = [i+direction, j+1-direction]
    
    # Build the environment
    if direction2 == 1:
        env.build(site2[0], site2[1], direction)
    else:
        env.build(site1[0], site1[1], direction)
    
    # Fetch the current tensors
    A1 = copy.deepcopy(env.psi.tensors[site1[0]][site1[1]])
    A2 = copy.deepcopy(env.psi.tensors[site2[0]][site2[1]])
    shape1 = shape(A1)
    shape2 = shape(A2)
    
    # Calculate the norm of the updated state
    gate2 = contract(dag(gate), gate, 0, 0)
    gate2 = trace(gate2, 1, 4)
    gate2 = permute(gate2, 1, 2)
    normGate = calculateOverlap(env, A1, A2, gate2, direction2)
    
    # Contract tensors with each other and gate
    if direction == 0:
        prod = contract(A1, A2, 3, 0)
    else:
        prod = contract(A1, A2, 2, 1)
    prod = contract(prod, gate, 3, 1)
    prod = trace(prod, 6, 9)
    prod = permute(prod, 6, 3)
    
    # Split the gate using SVD
    prod, cmb1 = combineIdxs(prod, [0, 1, 2, 3])
    prod, cmb2 = combineIdxs(prod, [0, 1, 2, 3])
    U, S, V = svd(prod, maxdim=maxdim)
    
    # Move singular values left and reshape into the correct form
    U = contract(U, S, 1, 0)
    if direction == 0:
        A1 = permute(U, 1, 0)
        A1 = reshape(A1, (shape(S)[0], shape1[0], shape1[1], shape1[2], shape1[4]))
        A1 = permute(A1, 0, 3)
        A2 = reshape(V, (shape(S)[1], shape2[1], shape2[2], shape2[3], shape2[4]))
    else:
        A1 = permute(U, 1, 0)
        A1 = reshape(A1, (shape(S)[0], shape1[0], shape1[1], shape1[3], shape1[4]))
        A1 = permute(A1, 0, 2)
        A2 = reshape(V, (shape(S)[1], shape2[0], shape2[2], shape2[3], shape2[4]))
        A2 = permute(A2, 0, 1)  
    
    
    # Fetch their new shapes
    shape1 = shape(A1)
    shape2 = shape(A2)
    
    # Flatten the tensors
    A1 = A1.flatten()
    A2 = A2.flatten()
    
    

    # Calculate the initial cost
    overlap = calculateOverlap(env, reshape(A1, shape1), reshape(A2, shape2),
                                 gate, direction2)
    norm = calculateNorm(env, reshape(A1, shape1), reshape(A2, shape2),
                         direction2)
    cost = [normGate + norm - overlap - np.conj(overlap)]
    
    k = 0
    converge = False
    while not converge:
        # Optimize first site
        def mv1(x):
            A = partialNorm(env, reshape(x, shape1), reshape(A2, shape2),
                            direction2, 0).flatten()
            return 0.5*(A+dag(A)).flatten()
        L1 = LinearOperator((np.size(A1), np.size(A1)), matvec=mv1, rmatvec=mv1)
        b1 = partialOverlap(env, reshape(A1, shape1), reshape(A2, shape2),
                            gate, direction2, 0).flatten()
        b1 = 0.5*(b1+dag(b1))
        #A1, info = cg(L1, b1, A1, maxiter=100, tol=10**-10)
        A1 = scipy.sparse.linalg.lsmr(L1, b1, maxiter=100, atol=10**-10,
                                      btol=10**-10, x0=A1)[0]
        
        # Optimize the second
        def mv2(y):
            A = partialNorm(env, reshape(A1, shape1), reshape(y, shape2),
                            direction2, 1).flatten()
            return 0.5*(A+dag(A)).flatten()
        L2 = LinearOperator((np.size(A2), np.size(A2)), matvec=mv2, rmatvec=mv2)
        b2 = partialOverlap(env, reshape(A1, shape1), reshape(A2, shape2),
                            gate, direction2, 1).flatten()
        b2 = 0.5*(b2+dag(b2))
        #A2, info = cg(L2, b2, A2, maxiter=100, tol=10**-10)
        A2 = scipy.sparse.linalg.lsmr(L2, b2, maxiter=100, atol=10**-10,
                                      btol=10**-10, x0=A2)[0]
        
        # Calculate the overlap and norms
        overlap = calculateOverlap(env, reshape(A1, shape1), reshape(A2, shape2),
                                 gate, direction2)
        norm = calculateNorm(env, reshape(A1, shape1), reshape(A2, shape2),
                             direction2)
        cost.append(normGate + norm - overlap - np.conj(overlap))
        
        # Check to see if converged
        k += 1
        if np.real(cost[-1]) > 10**-12:
            converge = np.real((cost[-1] - cost[-2]) / (cost[-1])) <= tol
        else:
            converge = True
        converge = converge or k >= maxiter
        if k == maxiter:
            print("Max iterations reached")
    
    # Update the tensors
    if np.max(np.abs(A1)) > 10**-10 and np.max(np.abs(A2)) > 10**-10:
        env.psi.tensors[site1[0]][site1[1]] = reshape(A1, shape1)*norm**(-0.5)
        env.psi.tensors[site2[0]][site2[1]] = reshape(A2, shape2)
    else:
        print("Optimization failed: using old tensors.")
        print(site1, site1, direction)
        norm = normGate
    
    # Build the environment
    if direction2 == 1:
        env.build(site1[0], site1[1], direction)
    else:
        env.build(site2[0], site2[1], direction)
    return norm, cost[-1] 
    


def calculateOverlap(env, A1, A2, gate, direction2=0):
    """
    Calculate the overlap which updated tensors.

    Parameters
    ----------
    env : environment
    A1 : np.ndarray
        First tensor in optimization.
    A2 : np.ndarray
        Second tensor in optimization.

    Returns
    -------
    prod : number
        The norm.

    """
    # Fetch the blocks
    left = env.leftBlock(env.center-1)
    right = env.rightBlock(env.center+1)
    leftMPS = env.leftMPSBlock(env.center2-1-direction2)
    rightMPS = env.rightMPSBlock(env.center2+2-direction2)
    
    # Get the bMPO tensors
    Mleft1 = left.tensors[env.center2-direction2]
    Mleft2 = left.tensors[env.center2+1-direction2]
    Mright1 = right.tensors[env.center2-direction2]
    Mright2 = right.tensors[env.center2+1-direction2]
    
    # Get the site tensors
    if env.dir == 0:
        B1 = env.psi.tensors[env.center][env.center2-direction2]
        B2 = env.psi.tensors[env.center][env.center2+1-direction2]
    else:
        B1 = env.psi.tensors[env.center2-direction2][env.center]
        B2 = env.psi.tensors[env.center2+1-direction2][env.center]
    
    # Do the contractions
    if env.dir == 0:
        prod = contract(leftMPS, Mleft1, 0, 0)
        prod = contract(prod, dag(A1), 0, 0)
        prod = trace(prod, 2, 5)
        prod = contract(prod, gate, 6, 0)
        prod = contract(prod, B1, 0, 0)
        prod = trace(prod, 1, 8)
        prod = trace(prod, 4, 9)
        prod = contract(prod, Mright1, 0, 0)
        prod = trace(prod, 1, 7)
        prod = trace(prod, 4, 6)
        prod = contract(prod, Mleft2, 0, 0)
        prod = contract(prod, dag(A2), 0, 0)
        prod = trace(prod, 4, 7)
        prod = trace(prod, 0, 8)
        prod = contract(prod, B2, 1, 0)
        prod = trace(prod, 2, 6)
        prod = trace(prod, 0, 7)
        prod = contract(prod, Mright2, 0, 0)
        prod = trace(prod, 1, 5)
        prod = trace(prod, 2, 4)
    else:
        prod = contract(leftMPS, Mleft1, 0, 0)
        prod = contract(prod, dag(A1), 0, 1)
        prod = trace(prod, 2, 5)
        prod = contract(prod, gate, 6, 0)
        prod = contract(prod, B1, 0, 1)
        prod = trace(prod, 1, 8)
        prod = trace(prod, 4, 9)
        prod = contract(prod, Mright1, 0, 0)
        prod = trace(prod, 2, 7)
        prod = trace(prod, 5, 6)
        prod = contract(prod, Mleft2, 0, 0)
        prod = contract(prod, dag(A2), 0, 1)
        prod = trace(prod, 4, 7)
        prod = trace(prod, 0, 8)
        prod = contract(prod, B2, 1, 1)
        prod = trace(prod, 2, 6)
        prod = trace(prod, 0, 7)
        prod = contract(prod, Mright2, 0, 0)
        prod = trace(prod, 2, 5)
        prod = trace(prod, 3, 4)
        
    # Contract with right block
    prod = contract(prod, rightMPS, 0, 0)
    prod = trace(prod,2, 5)
    prod = trace(prod,0, 2)
    prod = trace(prod,0, 1)
    return prod.item()


def calculateNorm(env, A1, A2, direction2=0):
    """
    Calculate the norm which updated tensors.

    Parameters
    ----------
    env : environment
    A1 : np.ndarray
        First tensor in optimization.
    A2 : np.ndarray
        Second tensor in optimization.
    gate : np.ndarray
        Two-site gate.

    Returns
    -------
    prod : number
        The overlap.

    """
    
    # Fetch the blocks
    left = env.leftBlock(env.center-1)
    right = env.rightBlock(env.center+1)
    leftMPS = env.leftMPSBlock(env.center2-1-direction2)
    rightMPS = env.rightMPSBlock(env.center2+2-direction2)
    
    As = [A1, A2]
    prod = copy.deepcopy(leftMPS)
    # Loop through twice, growing the block
    for i in range(2):
        # Fetch tensors
        A = As[i]
        Mleft = left.tensors[env.center2+i-direction2]
        Mright = right.tensors[env.center2+i-direction2]
        
        # Contract with middle
        if env.dir == 0:
            prod = contract(prod, Mleft, 0, 0)
            prod = contract(prod, dag(A), 0, 0)
            prod = trace(prod, 2, 5)
            prod = contract(prod, A, 0, 0)
            prod = trace(prod, 1, 6)
            prod = trace(prod, 4, 7)
            prod = contract(prod, Mright, 0, 0)
            prod = trace(prod, 1, 5)
            prod = trace(prod, 2, 4)
        else:
            prod = contract(prod, Mleft, 0, 0)
            prod = contract(prod, dag(A), 0, 1)
            prod = trace(prod, 2, 5)
            prod = contract(prod, A, 0, 1)
            prod = trace(prod, 1, 6)
            prod = trace(prod, 4, 7)
            prod = contract(prod, Mright, 0, 0)
            prod = trace(prod, 2, 5)
            prod = trace(prod, 3, 4)
        
    # Contract with right blocks
    prod = contract(prod, rightMPS, 0, 0)
    prod = trace(prod, 2, 5)
    prod = trace(prod, 0, 2)
    prod = trace(prod, 0, 1)
    
    return prod.item()
    
    
def partialNorm(env, A1, A2, direction2=0, site=0):
    """
    Calculate the partial norm which updated tensors.

    Parameters
    ----------
    env : environment
    A1 : np.ndarray
        First tensor in optimization.
    A2 : np.ndarray
        Second tensor in optimization.
    site : bool, optional
        The site which is being optimized, 0 = first, 1 = second.
        The default is 0.

    Returns
    -------
    prod : np.ndarray
        The norm matrix.

    """
    
    # Fetch the blocks
    left = env.leftBlock(env.center-1)
    right = env.rightBlock(env.center+1)
    leftMPS = env.leftMPSBlock(env.center2-1-direction2)
    rightMPS = env.rightMPSBlock(env.center2+2-direction2)
    
    # Get the bMPO tensors
    Mleft1 = left.tensors[env.center2-direction2]
    Mleft2 = left.tensors[env.center2+1-direction2]
    Mright1 = right.tensors[env.center2-direction2]
    Mright2 = right.tensors[env.center2+1-direction2]
    
    # Expand the relevent blocks
    if site == 1:
        # Expand the left block
        if env.dir == 0:
            leftMPS = contract(leftMPS, Mleft1, 0, 0)
            leftMPS = contract(leftMPS, dag(A1), 0, 0)
            leftMPS = trace(leftMPS, 2, 5)
            leftMPS = contract(leftMPS, A1, 0, 0)
            leftMPS = trace(leftMPS, 1, 6)
            leftMPS = trace(leftMPS, 4, 7)
            leftMPS = contract(leftMPS, Mright1,  0, 0)
            leftMPS = trace(leftMPS, 1, 5)
            leftMPS = trace(leftMPS, 2, 4)
        else:
            leftMPS = contract(leftMPS, Mleft1, 0, 0)
            leftMPS = contract(leftMPS, dag(A1), 0, 1)
            leftMPS = trace(leftMPS, 2, 5)
            leftMPS = contract(leftMPS, A1, 0, 1)
            leftMPS = trace(leftMPS, 1, 6)
            leftMPS = trace(leftMPS, 4, 7)
            leftMPS = contract(leftMPS, Mright1, 0, 0)
            leftMPS = trace(leftMPS, 2, 5)
            leftMPS = trace(leftMPS, 3, 4)
        Mleft = Mleft2
        Mright = Mright2
        A = A2
    else:
        # Expand the right block
        if env.dir == 0:
            rightMPS = contract(Mright2, rightMPS, 3, 3)
            rightMPS = contract(A2, rightMPS, 3, 5)
            rightMPS = trace(rightMPS, 2, 6)
            rightMPS = contract(dag(A2), rightMPS, 3, 6)
            rightMPS = trace(rightMPS, 2, 8)
            rightMPS = trace(rightMPS, 2, 5)
            rightMPS = contract(Mleft2, rightMPS, 3, 5)
            rightMPS = trace(rightMPS, 2, 6)
            rightMPS = trace(rightMPS, 1, 3)
        else:
            rightMPS = contract(Mright2, rightMPS, 3, 3)
            rightMPS = contract(A2, rightMPS, 2, 5)
            rightMPS = trace(rightMPS, 2, 6)
            rightMPS = contract(dag(A2), rightMPS, 2, 6)
            rightMPS = trace(rightMPS, 2, 8)
            rightMPS = trace(rightMPS, 2, 5)
            rightMPS = contract(Mleft2, rightMPS, 3, 5)
            rightMPS = trace(rightMPS, 2, 5)
            rightMPS = trace(rightMPS, 1, 2)
        Mleft = Mleft1
        Mright = Mright1
        A = A1
    
    # Contract the left block with middle sites and then right block
    if env.dir == 0:
        prod = contract(leftMPS, Mleft, 0, 0)
        prod = contract(prod, A, 1, 0)
        prod = trace(prod, 3, 5)
        prod = contract(prod, Mright, 1, 0)
        prod = trace(prod, 3, 7)
        prod = contract(prod, rightMPS, 2, 0)
        prod = trace(prod, 5, 8)
        prod = trace(prod, 2, 6)
    else:
        prod = contract(leftMPS, Mleft, 0, 0)
        prod = contract(prod, A, 1, 1)
        prod = trace(prod, 3, 5)
        prod = contract(prod, Mright, 1, 0)
        prod = trace(prod, 4, 7)
        prod = contract(prod, rightMPS, 2, 0)
        prod = trace(prod, 5, 8)
        prod = trace(prod, 2, 6)
    
    # Reshape into the correct form
    if env.dir == 0:
        prod = permute(prod, 2)
    else:
        prod = permute(prod, 0, 1)
        prod = permute(prod, 3, 4)
        prod = permute(prod, 2)
    
    return prod

def partialOverlap(env, A1, A2, gate, direction2=0, site=0):
    """
    Calculate the partial overlap after a trotter gate is applied.

    Parameters
    ----------
    env : environment
    A1 : np.ndarray
        First tensor in optimization.
    A2 : np.ndarray
        Second tensor in optimization.
    gate : np.ndarray
        Two-site gate.
    site : bool, optional
        The site which is being optimized, 0 = first, 1 = second.
        The default is 0.

    Returns
    -------
    prod : np.ndarray
        The overlap vector.

    """
    
    # Fetch the blocks
    left = env.leftBlock(env.center-1)
    right = env.rightBlock(env.center+1)
    leftMPS = env.leftMPSBlock(env.center2-1-direction2)
    rightMPS = env.rightMPSBlock(env.center2+2-direction2)
    
    # Get the bMPO tensors
    Mleft1 = left.tensors[env.center2-direction2]
    Mleft2 = left.tensors[env.center2+1-direction2]
    Mright1 = right.tensors[env.center2-direction2]
    Mright2 = right.tensors[env.center2+1-direction2]
    
    # Do the whole contractions for each of the four cases
    if env.dir == 0 and site == 1:
        prod = contract(leftMPS, Mleft1, 0, 0)
        prod = contract(prod, dag(A1), 0, 0)
        prod = trace(prod, 2, 5)
        prod = contract(prod, gate, 6, 0)
        prod = contract(prod, env.psi.tensors[env.center][env.center2-direction2], 0, 0)
        prod = trace(prod, 1, 8)
        prod = trace(prod, 4, 9)
        prod = contract(prod, Mright1, 0, 0)
        prod = trace(prod, 1, 7)
        prod = trace(prod, 4, 6)
        prod = contract(prod, Mright2, 5, 0)
        prod = contract(prod, env.psi.tensors[env.center][env.center2+1-direction2], 4, 0)
        prod = trace(prod, 5, 8)
        prod = trace(prod, 3, 8)
        prod = contract(prod, Mleft2, 0, 0)
        prod = trace(prod, 4, 7)
        prod = contract(prod, rightMPS, 6, 0)
        prod = trace(prod, 3, 8)
        prod = trace(prod, 3, 6)
        prod = permute(prod, 2, 3)
        prod = permute(prod, 1)
    elif env.dir == 0 and site == 0:
        prod = contract(Mright2, rightMPS, 3, 3)
        prod = contract(env.psi.tensors[env.center][env.center2+1-direction2], prod, 3, 5)
        prod = trace(prod, 2, 6)
        prod = contract(gate, prod, 3, 2)
        prod = contract(dag(A2), prod, 3, 8)
        prod = trace(prod, 2, 10)
        prod = trace(prod, 2, 5)
        prod = contract(Mleft2, prod, 3, 7)
        prod = trace(prod, 2, 8)
        prod = trace(prod, 1, 3)
        prod = contract(Mright1, prod, 3, 5)
        prod = contract(env.psi.tensors[env.center][env.center2-direction2], prod, 3, 7)
        prod = trace(prod, 2, 6)
        prod = trace(prod, 2, 8)
        prod = contract(Mleft1, prod, 3, 4)
        prod = trace(prod, 2, 4)
        prod = contract(leftMPS, prod, 3, 3)
        prod = trace(prod, 0, 3)
        prod = trace(prod, 1, 3)
    elif env.dir == 1 and site == 1:
        prod = contract(leftMPS, Mleft1, 0, 0)
        prod = contract(prod, dag(A1), 0, 1)
        prod = trace(prod, 2, 5)
        prod = contract(prod, gate, 6, 0)
        prod = contract(prod, env.psi.tensors[env.center2-direction2][env.center], 0, 1)
        prod = trace(prod, 1, 8)
        prod = trace(prod, 4, 9)
        prod = contract(prod, Mright1, 0, 0)
        prod = trace(prod, 2, 7)
        prod = trace(prod, 5, 6)
        prod = contract(prod, Mright2, 5, 0)
        prod = contract(prod, env.psi.tensors[env.center2+1-direction2][env.center], 4, 1)
        prod = trace(prod, 5, 9)
        prod = trace(prod, 3, 8)
        prod = contract(prod, Mleft2, 0, 0)
        prod = trace(prod, 4, 7)
        prod = contract(prod, rightMPS, 6, 0)
        prod = trace(prod, 3, 8)
        prod = trace(prod, 3, 6)
        prod = permute(prod, 1)
        prod = permute(prod, 2, 0)
        prod = permute(prod, 3, 2)
    elif env.dir == 1 and site == 0:
        prod = contract(Mright2, rightMPS, 3, 3)
        prod = contract(env.psi.tensors[env.center2+1-direction2][env.center], prod, 2, 5)
        prod = trace(prod, 2, 6)
        prod = contract(gate, prod, 3, 2)
        prod = contract(dag(A2), prod, 2, 8)
        prod = trace(prod, 2, 10)
        prod = trace(prod, 2, 5)
        prod = contract(Mleft2, prod, 3, 7)
        prod = trace(prod, 2, 7)
        prod = trace(prod, 1, 2)
        prod = contract(Mright1, prod, 3, 5)
        prod = contract(env.psi.tensors[env.center2-direction2][env.center], prod, 2, 7)
        prod = trace(prod, 2, 6)
        prod = trace(prod, 2, 8)
        prod = contract(Mleft1, prod, 3, 4)
        prod = trace(prod, 2, 3)
        prod = contract(leftMPS, prod, 3, 3)
        prod = trace(prod, 0, 3)
        prod = trace(prod, 1, 3)
        prod = permute(prod, 0, 1)
        prod = permute(prod, 2, 3)
    return prod
    
    
        