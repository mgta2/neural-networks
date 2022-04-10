# This file contains the NeuralNetwork class.

import numpy
import scipy.special

# Activation Function: Chosen here to be the Sigmoid
def act_func(x):
    return scipy.special.expit(x)

# Neural Network Class
class NeuralNetwork:
    
    def __init__(self, network_shape, learning_rate):
        self.network_shape = network_shape
        self.depth = len(network_shape)
        self.learning_rate = learning_rate
        self.weights = [None]*(self.depth-1)
        
        # Sample from normal distribution ~ N(0, 1/sqrt(num_nodes_in_given_level)) to obtain random initial weights.
        for i in range(self.depth-1):
            self.weights[i] = numpy.random.normal(0.0, pow(self.network_shape[i], -0.5), (self.network_shape[i+1], self.network_shape[i]))
    
    
    def print_weights(self):
        print(self.weights)
    
    
    def train(self, inputs, targets):
        
        if type(inputs) == list:
            inputs = numpy.array(inputs, ndmin=2).T
        if type(targets) == list:
            targets = numpy.array(targets, ndmin=2).T
        
        # Calculate the outputs.
        outputs = [None]*self.depth
        outputs[0] = inputs
        for i in range(self.depth-1):
            outputs[i+1] = act_func( numpy.dot(self.weights[i], outputs[i]) )
        
        # Calculate the errors.
        errors = [None]*(self.depth-1)
        errors[-1] = targets - outputs[-1]
        for i in range(self.depth-3, -1, -1):
            errors[i] = numpy.dot(self.weights[i+1].T, errors[i+1])
        
        # Update the weights.
        for i in range(self.depth-2, -1, -1):
            self.weights[i] += self.learning_rate * numpy.dot( (errors[i] * outputs[i+1] * (1 - outputs[i+1])), outputs[i].T )
        
    
    def query(self, inputs):
        
        if type(inputs) == list:
            inputs = numpy.array(inputs, ndmin=2).T
        
        for i in range(self.depth-1):
            outputs = act_func( numpy.dot(self.weights[i], inputs) )
            inputs = outputs
        
        return outputs
