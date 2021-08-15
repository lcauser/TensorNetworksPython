"""
    PEPS are difficult to contract, and require approximations to efficently
    do so. This class builds an environment around sites on a PEPS to update,
    or optimize tensors on a PEPS.
"""

# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.sitetypes import sitetypes
from tensornetworks.structures.opList2d import opList2d
from tensornetworks.structures.peps import peps
from tensornetworks.structures.mpo import bMPO
from tensornetworks.algorithms.vbMPO import vbMPO


class environment:
    
    def __init__(self, psi : peps, chi=0, nsites=2):
        # Store the information
        self.psi = psi
        if chi != 0:
            self.chi = chi
        else:
            self.chi = 5*psi.maxBondDim()**2
        self.nsites = nsites
        
        # Store bMPO blocks
        self.dir = 0 # 0 = vertical, 1 = horizontal
        self.blocks = [0] * self.psi.length[0]
        self.center = None
        self.blocks2 = [0] * self.psi.length[1]
        self.center2 = None
        self.build(0, 0)
        
    
    def leftBlock(self, idx):
        if idx < 0:
            return bMPO(self.psi.length[1])
        else:
            return self.blocks[idx]
    
    def rightBlock(self, idx):
        if idx > self.psi.length[self.dir] - 1:
            return bMPO(self.psi.length[1])
        else:
            return self.blocks[idx]
        
    def leftMPSBlock(self, idx):
        if idx < 0:
            return ones((1, 1, 1, 1))
        else:
            return self.blocks2[idx]
    
    def rightMPSBlock(self, idx):
        if idx > self.psi.length[1-self.dir] - 1:
            return ones((1, 1, 1, 1))
        else:
            return self.blocks2[idx]
    
    
    def buildUp(self, idx):
        # Retrieve the previous down block
        if idx == self.psi.length[0] - 1:
            prev = bMPO(self.psi.length[1])
        else:
            prev = copy.deepcopy(self.blocks[idx+1])
        
        # Loop through each tensor in the next row
        for i in range(self.psi.length[1]):
            # Retrieve the tensors
            M = prev.tensors[i]
            A = self.psi.tensors[idx][i]
            
            # Contract with both sites
            prod = contract(M, dag(A), 1, 2)
            prod = contract(prod, A, 1, 2)
            prod = trace(prod, 5, 9)
            
            # Reshape into bMPO tensor
            prod, cmb = combineIdxs(prod, [0, 2, 5])
            prod = permute(prod, 5, 0)
            prod, cmb = combineIdxs(prod, [1, 3, 5])
            
            # Update the tensor
            prev.tensors[i] = prod
        
        # Apply variational sweeps to limit the bond dimension
        prev = vbMPO(prev, self.chi)
        
        # Save the block
        self.blocks[idx] = prev
        return None
        
            
    def buildDown(self, idx):
        # Retrieve the previous down block
        if idx == 0:
            prev = bMPO(self.psi.length[1])
        else:
            prev = copy.deepcopy(self.blocks[idx-1])
        
        # Loop through each tensor in the next row
        for i in range(self.psi.length[1]):
            # Retrieve the tensors
            M = prev.tensors[i]
            A = self.psi.tensors[idx][i]
            
            # Contract with both sites
            prod = contract(M, dag(A), 1, 1)
            prod = contract(prod, A, 1, 1)
            prod = trace(prod, 5, 9)
            
            # Reshape into bMPO tensor
            prod, cmb = combineIdxs(prod, [0, 2, 5])
            prod = permute(prod, 5, 0)
            prod, cmb = combineIdxs(prod, [1, 3, 5])
            
            # Update the tensor
            prev.tensors[i] = prod
        
        # Apply variational sweeps to limit the bond dimension
        prev = vbMPO(prev, self.chi)
        
        # Save the block
        self.blocks[idx] = prev
        return None
    
    
    def buildRight(self, idx):
        # Retrieve the previous down block
        if idx == 0:
            prev = bMPO(self.psi.length[0])
        else:
            prev = copy.deepcopy(self.blocks[idx-1])
        
        # Loop through each tensor in the next row
        for i in range(self.psi.length[0]):
            # Retrieve the tensors
            M = prev.tensors[i]
            A = self.psi.tensors[i][idx]
            
            # Contract with both sites
            prod = contract(M, dag(A), 1, 0)
            prod = contract(prod, A, 1, 0)
            prod = trace(prod, 5, 9)
            
            # Reshape into bMPO tensor
            prod, cmb = combineIdxs(prod, [0, 2, 5])
            prod = permute(prod, 5, 0)
            prod, cmb = combineIdxs(prod, [1, 2, 4])
            
            # Update the tensor
            prev.tensors[i] = prod
        
        
        # Apply variational sweeps to limit the bond dimension
        prev = vbMPO(prev, self.chi)
        
        # Save the block
        self.blocks[idx] = prev
        return None
    
    
    def buildLeft(self, idx):
        # Retrieve the previous down block
        if idx == self.psi.length[1] - 1:
            prev = bMPO(self.psi.length[0])
        else:
            prev = copy.deepcopy(self.blocks[idx+1])
        
        # Loop through each tensor in the next row
        for i in range(self.psi.length[0]):
            # Retrieve the tensors
            M = prev.tensors[i]
            A = self.psi.tensors[i][idx]
            
            # Contract with both sites
            prod = contract(M, dag(A), 1, 3)
            prod = contract(prod, A, 1, 3)
            prod = trace(prod, 5, 9)
            
            # Reshape into bMPO tensor
            prod, cmb = combineIdxs(prod, [0, 3, 6])
            prod = permute(prod, 5, 0)
            prod, cmb = combineIdxs(prod, [1, 3, 5])
            
            # Update the tensor
            prev.tensors[i] = prod
        
        
        # Apply variational sweeps to limit the bond dimension
        prev = vbMPO(prev, self.chi)
        
        # Save the block
        self.blocks[idx] = prev
        return None
    
    
    def buildMPSRight(self, idx):
        # Fetch the left blocks
        if idx == 0:
            left = ones((1, 1, 1, 1))
        else:
            left = self.blocks2[idx-1]
        
        # Fetch the new blocks
        if self.center == 0:
            M1 = ones((1, 1, 1, 1))
        else:
            M1 = self.blocks[self.center-1].tensors[idx]
        if self.center == self.psi.length[0]- 1:
            M2 = ones((1, 1, 1, 1))
        else:
            M2 = self.blocks[self.center+1].tensors[idx]            
        if self.dir == 0:
            A = self.psi.tensors[self.center][idx]
        else:
            A = self.psi.tensors[idx][self.center]
        
        # Contract them all to grow the block
        if self.dir == 0:
            prod = contract(left, M1, 0, 0)
            prod = contract(prod, dag(A), 0, 0)
            prod = trace(prod, 2, 5)
            prod = contract(prod, A, 0, 0)
            prod = trace(prod, 1, 6)
            prod = trace(prod, 4, 7)
            prod = contract(prod, M2, 0, 0)
            prod = trace(prod, 1, 5)
            prod = trace(prod, 2, 4)
        else:
            prod = contract(left, M1, 0, 0)
            prod = contract(prod, dag(A), 0, 1)
            prod = trace(prod, 2, 5)
            prod = contract(prod, A, 0, 1)
            prod = trace(prod, 1, 6)
            prod = trace(prod, 4, 7)
            prod = contract(prod, M2, 0, 0)
            prod = trace(prod, 2, 5)
            prod = trace(prod, 3, 4)
        
        self.blocks2[idx] = prod
    
    
    def buildMPSLeft(self, idx):
        # Fetch the right blocks
        if idx == len(self.blocks2)-1:
            right = ones((1, 1, 1, 1))
        else:
            right = self.blocks2[idx+1]
        
        # Fetch the new blocks
        if self.center == 0:
            M1 = ones((1, 1, 1, 1))
        else:
            M1 = self.blocks[self.center-1].tensors[idx]
        if self.center == self.psi.length[0]- 1:
            M2 = ones((1, 1, 1, 1))
        else:
            M2 = self.blocks[self.center+1].tensors[idx]            
        if self.dir == 0:
            A = self.psi.tensors[self.center][idx]
        else:
            A = self.psi.tensors[idx][self.center]
            
        # Contract them all to grow the block
        if self.dir == 0:
            prod = contract(M2, right, 3, 3)
            prod = contract(A, prod, 3, 5)
            prod = trace(prod, 2, 6)
            prod = contract(dag(A), prod, 3, 6)
            prod = trace(prod, 2, 8)
            prod = trace(prod, 2, 5)
            prod = contract(M1, prod, 3, 5)
            prod = trace(prod, 2, 6)
            prod = trace(prod, 1, 3)
        else:
            prod = contract(M2, right, 3, 3)
            prod = contract(A, prod, 2, 5)
            prod = trace(prod, 2, 6)
            prod = contract(dag(A), prod, 2, 6)
            prod = trace(prod, 2, 8)
            prod = trace(prod, 2, 5)
            prod = contract(M1, prod, 3, 5)
            prod = trace(prod, 2, 5)
            prod = trace(prod, 1, 2)            
        
        self.blocks2[idx] = prod
            
    
    def build(self, idx1, idx2, direction=0):
        # Check if direction is correct
        if direction != self.dir:
            self.dir = direction
            self.blocks = [0] * self.psi.length[self.dir]
            self.center = None
            self.blocks2 = [0] * self.psi.length[1 - self.dir]
            self.center2 = None
        
        # Move the blocks
        rebuild = True
        if self.dir == 0:
            if self.center == None:
                for i in range(idx1):
                    self.buildDown(i)
                for i in range(self.psi.length[0] - 1 - idx1):
                    self.buildUp(self.psi.length[0]-1-i)
            elif self.center < idx1:
                for i in range(idx1 - self.center):
                    self.buildDown(self.center + i)
            elif self.center > idx1:
                for i in range(self.center - idx1):
                    self.buildUp(self.center - i)
            else:
                rebuild = False
            self.center = idx1
        else:
            if self.center == None:
                for i in range(idx2):
                    self.buildRight(i)
                for i in range(self.psi.length[1] - 1 - idx2):
                    self.buildLeft(self.psi.length[1]-1-i)
            elif self.center < idx2:
                for i in range(idx2 - self.center):
                    self.buildRight(self.center + i)
            elif self.center > idx2:
                for i in range(self.center - idx2):
                    self.buildLeft(self.center - i)
            else:
                rebuild = False
            self.center = idx2
        
        # Check to see if we should rebuild MPS blocks
        if rebuild == True:
            self.blocks2 = [0] * self.psi.length[1 - self.dir]
            self.center2 = None
        
        # Find the index to build to
        if self.dir == 0:
            idx = idx2
        else:
            idx = idx1
            
        # Build MPS-like blocks
        if self.center2 == None:
            for i in range(idx):
                self.buildMPSRight(i)
            for i in range(self.psi.length[1-self.dir]-1-idx):
                self.buildMPSLeft(self.psi.length[1-self.dir]-1-i)
        else:
            if self.center2 < idx:
                for i in range(idx - self.center2):
                    self.buildMPSRight(self.center2 + i)
            elif self.center2 > idx:
                for i in range(self.center2 - idx):
                    self.buildMPSLeft(self.center2 - i)
        self.center2 = idx
            
            
        return None
    
    
    def calculate(self):
        # Fetch the left and right blocks
        if self.center2 == 0:
            left = ones((1, 1, 1, 1))
        else:
            left = self.blocks2[self.center2-1]
        if self.center2 == len(self.blocks2)-1:
            right = ones((1, 1, 1, 1))
        else:
            right = self.blocks2[self.center2+1]
        
        # Fetch the PEPS tensor
        if self.dir == 0:
            A = self.psi.tensors[self.center][self.center2]
        else:
            A = self.psi.tensors[self.center2][self.center]
        
        # Fetch the bMPO tensors
        if self.center == 0:
            M1 = ones((1, 1, 1, 1))
        else:
            M1 = self.blocks[self.center-1].tensors[self.center2]
        if self.center == len(self.blocks)-1:
            M2 = ones((1, 1, 1, 1))
        else:
            M2 = self.blocks[self.center+1].tensors[self.center2]
        
        
        # Complete the contraction
        if self.dir == 0:
            # Horizontal
            prod = contract(left, M1, 0, 0)
            prod = contract(prod, dag(A), 0, 0)
            prod = trace(prod, 2, 5)
            prod = contract(prod, A, 0, 0)
            prod = trace(prod, 1, 6)
            prod = trace(prod, 4, 7)
            prod = contract(prod, M2, 0, 0)
            prod = trace(prod, 1, 5)
            prod = trace(prod, 2, 4)
        else:
            prod = contract(left, M1, 0, 0)
            prod = contract(prod, dag(A), 0, 1)
            prod = trace(prod, 2, 5)
            prod = contract(prod, A, 0, 1)
            prod = trace(prod, 1, 6)
            prod = trace(prod, 4, 7)
            prod = contract(prod, M2, 0, 0)
            prod = trace(prod, 2, 5)
            prod = trace(prod, 3, 4)
        
        # Contract with right block
        prod = contract(prod, right, 0, 0)
        prod = trace(prod, 2, 5)
        prod = trace(prod, 0, 2)
        prod = trace(prod, 0, 1)
        return prod.item()
                

def expectationOpEnv(env : environment, ops, site, direction=0):
    if not isinstance(ops, list):
        ops = [ops]
    if len(site) != 2:
        raise ValueError("The site must have two indexs.")
    
    # Build up the environment
    env.build(site[0], site[1], direction)
    
    # Fetch the left and right blocks
    if env.center2 == 0:
        left = ones((1, 1, 1, 1))
    else:
        left = env.blocks2[env.center2-1]
    if env.center2 >= len(env.blocks2)-len(ops):
        right = ones((1, 1, 1, 1))
    else:
        right = env.blocks2[env.center2+len(ops)]
    
    prod = copy.deepcopy(left)
    for i in range(len(ops)):
        # Fetch the PEPS tensor and apply the operator
        if env.dir == 0:
            A = env.psi.tensors[env.center][env.center2+i]
        else:
            A = env.psi.tensors[env.center2+i][env.center]
        A = contract(A, ops[i], 4, 1)
        
        # Fetch the bMPO tensors
        if env.center == 0:
            M1 = ones((1, 1, 1, 1))
        else:
            M1 = env.blocks[env.center-1].tensors[env.center2+i]
        if env.center == len(env.blocks)-1:
            M2 = ones((1, 1, 1, 1))
        else:
            M2 = env.blocks[env.center+1].tensors[env.center2+i]
            
        # Complete the contraction
        if env.dir == 0:
            # Horizontal
            prod = contract(prod, M1, 0, 0)
            prod = contract(prod, dag(A), 0, 0)
            prod = trace(prod, 2, 5)
            prod = contract(prod, A, 0, 0)
            prod = trace(prod, 1, 6)
            prod = trace(prod, 4, 7)
            prod = contract(prod, M2, 0, 0)
            prod = trace(prod, 1, 5)
            prod = trace(prod, 2, 4)
        else:
            prod = contract(prod, M1, 0, 0)
            prod = contract(prod, dag(A), 0, 1)
            prod = trace(prod, 2, 5)
            prod = contract(prod, A, 0, 1)
            prod = trace(prod, 1, 6)
            prod = trace(prod, 4, 7)
            prod = contract(prod, M2, 0, 0)
            prod = trace(prod, 2, 5)
            prod = trace(prod, 3, 4)
        
    # Contract with right block
    prod = contract(prod, right, 0, 0)
    prod = trace(prod, 2, 5)
    prod = trace(prod, 0, 2)
    prod = trace(prod, 0, 1)
    
    return prod.item()

def expectationOpListEnv(env : environment, ops : opList2d):
    expectations = [0] * len(ops.sites)
    # Loop through each row
    for i in range(env.psi.length[0]):
        # Loop through each site in the row
        for j in range(env.psi.length[1]):
            # Find the indexs where horizontal bonds are
            idxs = ops.siteIndexs([i, j], 0)
            
            # Loop through each index, and find the expectation
            for idx in idxs:
                opers = [ops.sitetype.op(op) for op in ops.ops[idx]]
                expectations[idx] = expectationOpEnv(env, opers, [i, j], 0)
                expectations[idx] *= ops.coeffs[idx]
    
    # Loop through each column
    for j in range(env.psi.length[1]):
        # Loop through each site in the row
        for i in range(env.psi.length[0]):
            # Find the indexs where horizontal bonds are
            idxs = ops.siteIndexs([i, j], 1)
            
            # Loop through each index, and find the expectation
            for idx in idxs:
                opers = [ops.sitetype.op(op) for op in ops.ops[idx]]
                expectations[idx] = expectationOpEnv(env, opers, [i, j], 1)
                expectations[idx] *= ops.coeffs[idx]
    
    return expectations
