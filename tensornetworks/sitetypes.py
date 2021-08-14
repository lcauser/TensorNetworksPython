"""
    Site types allow users to create custom lattice defintions, by giving names
    to local states and operators. It can then be used to generate tensor
    products, finding products and hermitians, etc.
"""

from tensornetworks.tensors import tensor, contract
import numpy as np

class sitetypes:
    
    def __init__(self, dim):
        """
        Define a lattice type, with states and operators.

        Parameters
        ----------
        dim : int
            Local physical dimensions.

        """
        # Save the dimension
        self.dim = dim
        
        # Create arrays to store states and operators
        self.stateNames = []
        self.states = []
        self.opNames = []
        self.ops = []
        self.opDags = []
        
        # A counter for new operators
        self.temp = 0
        
    
    def addState(self, name : str, A):
        """ Add a vector state to the sitetype.
        
        Parameters:
            name (str): the name of the site e.g. "up"
            A (Tensor): must have dimensions (dim).
        """
        
        # Check to see if the matrix is the correct type
        if np.shape(A) != (self.dim, ):
            print("This must be a tensor of dimensions ("+str(self.dim)+", ).")
            return False
        
        # Check to see if the name exists
        if name in self.stateNames:
            print("The name is already taken.")
            return False
        
        # Add to list
        self.stateNames.append(name)
        self.states.append(A)
    
    
    def state(self, name : str):
        """ Returns the tensor for the named state.
        
        Parameters:
            name (str): name of the state
        
        Returns:
            A (tensor): the tensor
        """
        
        # Check to see if name exists
        if not name in self.stateNames:
            print("The state is not defined.")
            return None
        
        # Find the idx
        return self.states[self.stateNames.index(name)]
    
    
    def addOp(self, name : str, A, hermitian = False):
        """ Add a operator to the sitetype.
        
        Parameters:
            name (str): the name of the operator e.g. "X"
            A (Tensor): must have dimensions (dim, dim).
        """
        
        # Check to see if the matrix is the correct type
        if np.shape(A) != (self.dim, self.dim):
            print("This must be a tensor of dimensions ("+str(self.dim)+","+str(self.dim)+").")
            return False
        
        # Check to see if the name exists
        if name in self.opNames:
            print("The name is already taken.")
            return False
        
        # Add to list
        self.opNames.append(name)
        self.ops.append(A)
        self.opDags.append(hermitian)
    
    
    def op(self, name : str):
        """ Returns the tensor for the operator.
        
        Parameters:
            name (str): name of the operator
        
        Returns:
            A (tensor): the tensor
        """
        
        # Check to see if name exists
        if not name in self.opNames:
            print("The operator is not defined.")
            return None
        
        # Find the idx
        return self.ops[self.opNames.index(name)]
    
    
    def dagger(self, name : str):
        """ Find the name of the hermitian conjugate of an operator.
        
        Parameters:
            name (str): the name of the operator e.g. "s+"
        
        Returns:
            conj (str): the name of the hermitian conjugate e.g. "s-"
        """
        conj = self.opDags[self.opNames.index(name)]
        if not conj in self.opNames:
            raise ValueError("The hermitian operator is not defined.")
        return conj
    
    
    def opProd(self, ops):
        """
        Take the product of local operators, and return the named operator.
        Defines a new operator if it cannot find an existing equivalent.

        Parameters
        ----------
        ops : list
            List of operators.

        Returns
        -------
        op : str
            Operator name

        """
        # Make sure it's a list
        if not isinstance(ops, list):
            ops = [ops]
            
        # Start the operator product with the identity, and contract operators
        prod = self.op("id")
        for op in ops:
            prod = contract(prod, self.op(op), 1, 0)
        
        # Check to see if the operator is defined in the op list
        for idx in range(len(self.ops)):
            if np.all(np.abs(prod - self.ops[idx]) < 10**-14):
                return self.opNames[idx]
        
        # Create a new operator
        opName = "temp" + str(self.temp)
        self.temp += 1
        self.addOp(opName, prod)
        return opName

