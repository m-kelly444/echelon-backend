import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class Variable:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name or ""
        self.derivative = None
        self.is_constant = False
        self.history = None
    
    def accumulate_derivative(self, derivative):
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += derivative
    
    def is_leaf(self):
        return self.history is None or self.history.last_fn is None
    
    def backward(self, derivative=None):
        if derivative is None:
            derivative = 1.0
        backpropagate(self, derivative)

class Context:
    def __init__(self, no_grad=False):
        self.no_grad = no_grad
        self.saved_values = ()
    
    def save_for_backward(self, *values):
        if self.no_grad:
            return
        self.saved_values = values
    
    @property
    def saved_tensors(self):
        return self.saved_values

def backpropagate(variable, deriv):
    if variable.is_leaf():
        variable.accumulate_derivative(deriv)
        return
    
    if variable.history is None:
        return
    
    for next_var, grad in variable.history.chain_rule(deriv):
        backpropagate(next_var, grad)

def central_difference(f, *vals, arg=0, epsilon=1e-6):
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    return (f(*vals1) - f(*vals2)) / (2.0 * epsilon)
