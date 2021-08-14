"""
    Operator lists store lists of many-body operators which can act on an MPS.
    It makes use of sitetypes to build operators efficiently. Also contains
    efficient ways of calculating multiple expectations simultanteously.
"""

# Imports
import numpy as np # Will be the main numerical resource
import copy # To make copies
from tensornetworks.tensors import *
from tensornetworks.sitetypes1d import sitetypes1d 
from tensornetworks.structures.mps import mps
from tensornetworks.structures.projMPS import projMPS
import numbers

class opList:
    
    def __init__(self, s : sitetypes1d, length):
        """
        Create a list of operators.

        Parameters
        ----------
        s : sitetypes1d
            Site types with operator names.
        length : int
            Length of lattice.

        """
        # Store information
        self.length = length
        self.sitetype = s
        self.ops = []
        self.sites = []
        self.coeffs = []
        return None
    
    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            opList = copy.deepcopy(self)
            for idx in range(len(opList.coeffs)):
                opList.coeffs[idx] *= other
            opList.sitetype = self.sitetype
        return opList
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    
    def __add__(self, other):
        if isinstance(other, opList):
            return addOpLists(self, other)
        raise ValueError("Type error.")
        
        
    def __radd__(self, other):
        return self.__add__(other)
    
    
    def add(self, ops, sites, coeff=1):
        """
        Add an operator to the list.

        Parameters
        ----------
        ops : list
            List of operator names
        sites : list
            List of sites they act on.
        coeff : number, optional
            coefficient of the operator. The default is 1.

        """
        if not isinstance(ops, list):
            ops = [ops]
        if not isinstance(sites, list):
            sites = [sites]
        
        # Check the length of ops and sites are the same
        if len(ops) != len(sites):
            raise ValueError("The operator and site list must be of equal size.")
        
        for op in ops:
            if op not in self.sitetype.opNames:
                raise ValueError("The operator is not defined.")
        
        for site in sites:
            if not isinstance(site, int):
                raise ValueError("The site must be an integer.")
        
        if not isinstance(coeff, numbers.Number):
            raise ValueError("The site must be a real or complex number.")
        
        # Sort the operator and site list
        ops = [op for _, op in sorted(zip(sites, ops))]
        sites = sorted(sites)
        
        # Add to the operator list
        self.ops.append(ops)
        self.sites.append(sites)
        self.coeffs.append(coeff)
        
        return None
    
    
    def siteRange(self):
        """
        Calculate the 'locality' of the list of operators i.e. the maximum 
        range of sites.

        Returns
        -------
        rng : int
            The maximum interaction range.

        """
        rng = 1
        for sites in self.sites:
            rng = max(rng, max(sites)-min(sites)+1)
        return rng
    
    
    def siteIndexs(self, site):
        """
        Return the indexs of all operators in the list which start at a site.

        Parameters
        ----------
        site : int
            First operator site.

        Returns
        -------
        idxs : list
            List of indexs

        """
        idxs = []
        for i in range(len(self.sites)):
            if min(self.sites[i]) == site:
                idxs.append(i)
        return idxs
    
    
    def toTensor(self, idx):
        """
        Create the tensor object for an operator in the list.

        Parameters
        ----------
        idx : int
            Operator index.

        Returns
        -------
        tensor: np.ndarray
            The tensor representation of the operator

        """
        # Fetch the relevent information and find the interaction range
        ops = self.ops[idx]
        sites = self.sites[idx]
        rng = min(self.siteRange(), self.length-min(sites))
        
        # Create the tensor through taking a tensor product. Insert identity
        # where no operator is given.
        prod = tensor((1, 1), [[1]])
        for site in range(rng):
            if min(sites)+site in sites:
                op = ops[site]
            else:
                op = "id"
            
            op = np.reshape(self.sitetype.op(op), (1, self.sitetype.dim,
                                                   self.sitetype.dim, 1))
            prod = contract(prod, op, len(np.shape(prod))-1, 0)
        prod = trace(prod, 0, len(np.shape(prod))-1)
        return self.coeffs[idx]*prod
        
    
    def siteTensor(self, site):
        """
        Calculate the tensor which acts from site to site + interaction range.

        Parameters
        ----------
        site : int
            First site.

        Returns
        -------
        tensor: np.ndarray
            Sum of all tensors starting at the site.

        """
        idxs = self.siteIndexs(site)
        if len(idxs) == 0:
            return False
        ten = 0
        for idx in idxs:
            ten += self.toTensor(idx)
        
        return ten


def addOpLists(opList1 : opList, opList2 : opList):
    """
    Add operator lists.

    Parameters
    ----------
    opList1 : opList
    opList2 : opList

    Returns
    -------
    opList3 : opList
        Summed operator lists.

    """
    if opList1.sitetype is not opList2.sitetype:
        raise ValueError("The operator lists must share the same site types.")
    opList3 = copy.deepcopy(opList1)
    opList3.sitetype = opList1.sitetype
    opList3.ops += copy.deepcopy(opList2.ops)
    opList3.sites += copy.deepcopy(opList2.sites)
    opList3.coeffs += copy.deepcopy(opList2.coeffs)
    
    return opList3
    
    
def opExpectation(ops : opList, psi : mps):
    """
    Calculate the expectation values of all operators in the list on an MPS.
    Does it efficiently by constructing the dot projection of an MPS,
    and reusing tensor contractions.

    Parameters
    ----------
    ops : opList
    psi : mps

    Returns
    -------
    expectations : list
        List of expectation values

    """
    # Create a projector of the dot product
    projDot = projMPS(psi, psi)
    
    # Loop through each site
    expectations = [0] * len(ops.ops)
    for site in range(psi.length):
        # Move the center of the projector
        projDot.moveCenter(site)
        
        # Find each operator which starts at the site
        idxs = ops.siteIndexs(site)
        
        for idx in idxs:
            # Find the range of the operator and fetch the blocks
            rng = max(ops.sites[idx]) - min(ops.sites[idx]) + 1
            site2 = site + rng - 1
            left = projDot.blocks[site-1] if site > 0 else np.ones((1, 1))
            right = projDot.blocks[site2+1] if site2 < psi.length-1 else np.ones((1, 1))
            

            # Loop through the middle sites, applying the operators
            for i in range(rng):
                # Fetch the site tensors
                A = psi.tensors[site + i]
                if site+i in ops.sites[idx]:
                    O = ops.sitetype.op(ops.ops[idx][ops.sites[idx].index(site+i)])
                else:
                    O = ops.sitetype.op("id")
                    
                left = contract(left, dag(A), 0, 0)
                left = contract(left, O, 1, 0)
                left = contract(left, A, 0, 0)
                left = trace(left, 1, 2)
                
            # Contract with right
            expectations[idx] = np.asscalar(trace(contract(left, right, 0, 0),
                                                  0, 1)) * ops.coeffs[idx]
    return expectations
            
        
def applyOp(psi, sitetype, ops, sites, coeff = 1):
    """
    Apply an operator to an MPS.

    Parameters
    ----------
    psi : mps
    sitetype : sitetypes1d
    ops : list
        List of operators
    sites : list
        List of sites
    coeff : number, optional
        coefficient of operator. The default is 1.

    Returns
    -------
    psi : mps
        MPS with operator applied to it.

    """
    # Move orthogonal center to first affected site
    psi.orthogonalize(sites[0])
    
    # Loop through each operator
    for i in range(len(ops)):
        # Fetch site tensors
        A = psi.tensors[sites[i]]
        O = sitetype.op(ops[i])
        
        # Apply operator and restore tensor
        A = contract(A, O, 1, 1)
        A = permute(A, 1)
        psi.tensors[sites[i]] = A
    
    # Move center to end and apply coefficent
    psi.orthogonalize(psi.length-1)
    #psi *= coeff
    return psi
        
    