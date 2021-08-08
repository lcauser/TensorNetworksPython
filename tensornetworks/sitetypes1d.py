from tensornetworks.tensors import tensor
import numpy as np

class sitetypes1d:
    
    def __init__(self, dim):
        self.dim = dim
        self.stateNames = []
        self.states = []
        self.opNames = []
        self.ops = []
        
    
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
    
    
    def addOp(self, name : str, A):
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

