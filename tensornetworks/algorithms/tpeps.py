"""
    Evolve a PEPS in time under some trotter gates, and use variational
    minimization to truncate, keeping bond dimensions small.
"""

import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.structures.environment import environment

def calculateOverlap(env, A1, A2, gate):
    # Fetch the blocks
    left = env.leftBlock(env.center-1)
    right = env.rightBlock(env.center+1)
    leftMPS = env.leftMPSBlock(env.center2-1)
    rightMPS = env.rightMPSBlock(env.center2+2)
    
    # Get the bMPO tensors
    Mleft1 = left.tensors[env.center2]
    Mleft2 = left.tensors[env.center2+1]
    Mright1 = right.tensors[env.center2]
    Mright2 = right.tensors[env.center2+1]
    
    # Get the site tensors
    if env.dir == 0:
        B1 = env.psi.tensors[env.center][env.center2]
        B2 = env.psi.tensors[env.center][env.center2+1]
    else:
        B1 = env.psi.tensors[env.center2][env.center]
        B2 = env.psi.tensors[env.center2+1][env.center]
    
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
        prod = trace(prod, 3, 6)
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


def calculateNorm(env, A1, A2):
    # Fetch the blocks
    left = env.leftBlock(env.center-1)
    right = env.rightBlock(env.center+1)
    leftMPS = env.leftMPSBlock(env.center2-1)
    rightMPS = env.rightMPSBlock(env.center2+2)
    
    As = [A1, A2]
    prod = copy.deepcopy(leftMPS)
    # Loop through twice, growing the block
    for i in range(2):
        # Fetch tensors
        A = As[i]
        Mleft = left.tensors[env.center2+i]
        Mright = right.tensors[env.center2+i]
        
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
    
    
def partialNorm(env, A1, A2):
    # Fetch the blocks
    left = env.leftBlock(env.center-1)
    right = env.rightBlock(env.center+1)
    leftMPS = env.leftMPSBlock(env.center2-1)
    rightMPS = env.rightMPSBlock(env.center2+2)
    
    # Get the bMPO tensors
    Mleft1 = left.tensors[env.center2]
    Mleft2 = left.tensors[env.center2+1]
    Mright1 = right.tensors[env.center2]
    Mright2 = right.tensors[env.center2+1]
    
    # Calculate partial norm

def partialOverlap(env, A1, A2):
    # Calculate the partial overlap
    
    
    
    
        