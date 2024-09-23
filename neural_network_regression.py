import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
import random

from optimizers import GradientDescentOptimizer, ConjugateGradientOptimizer, AdamOptimizer
from utils import eval_activation_func, eval_activation_func_gradient, eval_loss_func, unflatten_from_vector
from config import get_config

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=get_config()["debug_mode"])
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class GenericFeedforwardNeuralNetwork:
    """ Base class for a generic feedforward neural network.  
        
    Methods
    -----------
    print_layer_sizes():
        Print the sizes of each layer.
    print_layers():
        Print the weights of each layer.
    loss(y_true, y_pred):
        Calculate the loss.
    eval(x_in):
        Evaluate the result given the input x_in.
    print_layer_grad_sizes():
        Print the sizes of each layer's gradients.
    print_grad_layers():
        Print the gradients of each layer's weights and bias.
    zero_grad():
        Set the gradient of all layers to zero.
    train(X_train, y_train, n_epoch):
        Train the network. To be implemented by a derived class. 
    """
    
    def __init__(self, in_dim, hidden_dim_list, out_dim, 
            activation_func_list, optimizer_params, loss_func="mse", optimizer_name="gradient_descent"):
        """
        Parameters
        ----------
        in_dim : Integer. 
            The input dimension.
        hidden_dim_list: List of integer.
            The list of number of hidden neurons in each hidden layer. Perceptron mode is 
            not supported so if this variable is an empty list, then an error will be raised.
        out_dim : Integer. 
            The output dimension. 
        activation_func_list: List of string. 
            The list of the name of the activation function in each layer. 
        optimizer_params: Dictionary. 
            The dictionary storing the parameters. 
        loss_func: String.
            The name of the loss function to be used. Default is mean squared error.
        optimizer_name: String.
            The name of the optimizer to be used. Defaut is gradient descent.
        """
        
        self.in_dim = in_dim
        self.hidden_dim_list = hidden_dim_list
        self.out_dim = out_dim
        self.activation_func_list = activation_func_list
        self.loss_func = loss_func
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
      
        n_hidden_layer = len(hidden_dim_list)
        n_layer = n_hidden_layer+1 # Last layer is from last hidden layer to output

        layer_list = []
        in_curr_dim_list = []
        out_curr_dim_list = []

        if n_hidden_layer == 0:
            raise Exception("In class %s: Perceptron mode not supported." % (self.__class__.__name__))

        if n_hidden_layer > 0:
            # Multilayer perceptron (MLP)
            for layer_idx in range(n_layer):
                if layer_idx == 0:
                    # First layer is from input to first hidden layer
                    in_curr_dim = in_dim
                    out_curr_dim = hidden_dim_list[0]
                elif layer_idx == n_hidden_layer:
                    # Last layer is from last hidden layer to output
                    in_curr_dim = hidden_dim_list[-1]
                    out_curr_dim = out_dim
                else:
                    in_curr_dim = hidden_dim_list[layer_idx-1]
                    out_curr_dim = hidden_dim_list[layer_idx]
                try:
                    layer = {
                        "weight": np.random.normal(size=(out_curr_dim, in_curr_dim)),
                        "bias": np.random.normal(size=(out_curr_dim)),                    
                        "in_curr_dim": in_curr_dim,
                        "out_curr_dim": out_curr_dim,
                        "activation_func": activation_func_list[layer_idx],
                        "sensitivity": np.zeros((out_curr_dim)), # For backpropagation
                        "weight_grad": np.zeros((out_curr_dim, in_curr_dim)), # For backpropagation
                        "bias_grad": np.zeros((out_curr_dim)), # For backpropagation
                    }
                except:
                    set_trace()
                layer_list.append(layer)
                in_curr_dim_list.append(in_curr_dim)
                out_curr_dim_list.append(out_curr_dim)
        else:
            # Perceptron, so only one layer from input to output
            in_curr_dim = in_dim
            out_curr_dim = out_dim
            layer = {
                "weight": np.random.normal(size=(out_curr_dim, in_curr_dim)),
                "bias": np.random.normal(size=(out_curr_dim)),
                "in_curr_dim": in_curr_dim,
                "out_curr_dim": out_curr_dim,
                "activation_func": activation_func_list[0],
                "sensitivity": np.zeros((out_curr_dim)), # For backpropagation
                "weight_grad": np.zeros((out_curr_dim, in_curr_dim)), # For backpropagation
                "bias_grad": np.zeros((out_curr_dim)), # For backpropagation
            }

            layer_list.append(layer)
            in_curr_dim_list.append(in_curr_dim)
            out_curr_dim_list.append(out_curr_dim)

        self.n_hidden_layer = n_hidden_layer
        self.n_layer = n_layer
        self.layer_list = layer_list
        
    def print_layer_sizes(self):
        """ Print the sizes of each layer.

        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
        
        print("Printing layer sizes for debugging purpose")
        for layer_idx, layer in enumerate(self.layer_list):
            print("At layer %d" % (layer_idx+1))
            print("Weight: ")
            print(layer["weight"].shape)
            print("Bias: ")
            print(layer["bias"].shape)
    
    def print_layers(self):
        """ Print the weights of each layer.

        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
        
        print("Printing layers")
        for layer_idx, layer in enumerate(self.layer_list):
            print("At layer %d" % (layer_idx+1))
            print("Weight: ")
            print(layer["weight"])
            print("Bias: ")
            print(layer["bias"])
    
    def loss(self, y_true, y_pred):
        """ Calculate the loss.

        Parameters
        ---------------
        y_true: Numpy array.
            The ground truth array.
        y_pred: Numpy array.
            The prediction array.
        
        Returns
        ----------
        loss: Float.
            The loss value.
        """
        # set_trace()
        return eval_loss_func(y_true, y_pred, self.loss_func)
    

    def eval(self, x_in):
        """ Evaluate the result given the input x_in.

        Parameters
        ---------------
        x_in : Numpy array.
            Input, of size (n_features) where n_features is the number of features. 
        
        Returns
        ----------
        out: Numpy array.
            The output. 
        """
        
        curr_in = x_in
        out_list = []
        for layer_idx, layer in enumerate(self.layer_list):
            weight = layer["weight"]
            bias = layer["bias"]
            out_mat_1 = np.dot(weight, curr_in)
            out_mat = out_mat_1.squeeze() + bias.squeeze()
            if np.isscalar(out_mat):
                out_mat = np.array([[out_mat]])
            else:
                out_mat = out_mat[:, np.newaxis]

            out = eval_activation_func(out_mat, layer["activation_func"])
            # set_trace()
            if (np.size(out.shape) > 1):
                if (out.shape[0] > 1) and (out.shape[1] > 1):
                    set_trace()
                    raise Exception("EVAL ERROR!")
            
            curr_in = out     
            out_list.append(out)
        self.x_in = x_in
        self.out_list = out_list
        # set_trace()
        return out, out_list

    def print_layer_grad_sizes(self):
        """ Print the sizes of each layer's gradients.

        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
        
        print("Printing layer gradient sizes for debugging purpose")
        for layer_idx, layer in enumerate(self.layer_list):
            print("At layer %d" % (layer_idx+1))
            print("Weight: ")
            print(layer["weight_grad"].shape)
            print("Bias: ")
            print(layer["bias_grad"].shape)

    def print_grad_layers(self):
        """ Print the gradients of each layer's weights and bias.

        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
        
        print("Printing gradient of layers")
        for layer_idx, layer in enumerate(self.layer_list):
            print("At layer %d" % (layer_idx+1))
            print("Weight: ")
            print(layer["weight_grad"])
            print("Bias: ")
            print(layer["bias_grad"])

    def zero_grad(self):
        """ Set the gradient of all layers to zero.

        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
        
        for layer_idx, layer in enumerate(self.layer_list):
            layer["weight_grad"].fill(0)
            layer["bias_grad"].fill(0)

    def train(self, X_train, y_train, n_epoch):
        raise NotImplementedError


class RegressionNeuralNetwork(GenericFeedforwardNeuralNetwork):
    def __init__(self, in_dim, hidden_dim_list, out_dim, 
            activation_func_list, optimizer_params, loss_func="mse", optimizer_name="gradient_descent"):
        super(self.__class__, self).__init__(in_dim, hidden_dim_list, out_dim, 
            activation_func_list, optimizer_params, loss_func, optimizer_name)
  
        if self.optimizer_name == "gradient_descent":
            self.optimizer = GradientDescentOptimizer(optimizer_params)
        elif self.optimizer_name == "conjugate_gradient":
            self.optimizer = ConjugateGradientOptimizer(optimizer_params)
        elif self.optimizer_name == "Adam":
            self.optimizer = AdamOptimizer(optimizer_params)
        else:
            raise Exception("In class %s: Optimizer name %s not recognized" % (self.__class__.__name__, self.optimizer_name))


    def backprop(self, y_true):
        """ Backpropagation function.

        Parameters
        ---------------
        y_true: Numpy array. 
            The ground truth.
            
        Returns
        ----------
        None. 
        """
        
        # This function assumes you have called the eval function previously
        # then self.out_list will store the activation outputs at each layer
        # which will help with gradient calculation for backpropagation
        
        # First, calculate the sensitivity of the last layer based on the loss
        if self.loss_func == "mse":
            out = self.out_list[-1]
            fgrad = eval_activation_func_gradient(out, self.layer_list[-1]["activation_func"])
            # set_trace()
            Fdot = np.diag(fgrad)
            # set_trace()
            ss = (-2)*np.dot(Fdot, y_true-out)
            # set_trace()
            # For the Perceptron case
            if self.n_layer == 1:
                if np.size(ss.shape) == 1:
                    ss = ss[:, np.newaxis]
                    # set_trace()
            
            self.layer_list[-1]["sensitivity"] = ss  
        
        # Then, backpropagate to the previous layers
        for layer_idx in reversed(range(0, self.n_layer)):
            if layer_idx == (self.n_layer-1):
                continue
            out = self.out_list[layer_idx]
            fgrad = eval_activation_func_gradient(out, self.layer_list[layer_idx]["activation_func"])
            # set_trace()
            if (np.size(fgrad.shape) == 1) or (np.size(fgrad) == 1):
                Fdot = np.diag(fgrad)
            else:
                Fdot = np.diag(fgrad.squeeze())
            ss_next = self.layer_list[layer_idx+1]["sensitivity"]
            weight_next = self.layer_list[layer_idx+1]["weight"]
            ss = np.dot(Fdot, np.transpose(weight_next))
            ss = np.dot(ss, ss_next)
            
            if np.isscalar(ss):
                ss = np.array([[ss]])
            if np.size(ss.shape) == 1:
                ss = ss[:, np.newaxis]
            
            self.layer_list[layer_idx]["sensitivity"] = ss
            # set_trace()

    def update_gradient(self, y_true):
        """ Update the gradient. To be called after the backprop function.

        Parameters
        ---------------
        y_true: Numpy array. 
            The ground truth.
            
        Returns
        ----------
        None. 
        """
        
        for layer_idx, layer in enumerate(self.layer_list):
            ss = layer["sensitivity"]
            if layer_idx == 0:
                # try:
                weight_grad = np.dot(ss, np.transpose(self.x_in))
                # except:
                    # set_trace()
                bias_grad = ss
            else:
                weight_grad = np.dot(ss, np.transpose(self.out_list[layer_idx-1]))
                bias_grad = ss

            # Yeah I know this part looks horrible but here I'm just trying to study backprop so I just want to get the 
            # right result you know, can't concentrate on best code and neural network theory at the same time :(
            if np.size(weight_grad.shape) == 1:
                if np.size(layer["weight_grad"]) > 1 and (layer["weight_grad"].shape[0] == 1):
                    weight_grad = weight_grad[np.newaxis, :]
                else:
                    weight_grad = weight_grad[:, np.newaxis]
            layer["weight_grad"] += weight_grad 
            layer["bias_grad"] += bias_grad.squeeze()
             

    def train(self, X_train, y_train, n_epoch):
        """ Train the network.

        Parameters
        ---------------
        X_train: Numpy array. 
            The training data, of size (n_inst, n_features) where
            n_inst is the number of instances, and n_features is 
            the number of features.
        y_train: Numpy array.
            The ground truth, of size (n_inst).
        n_epoch: Integer.
            The number of epochs
        
        Returns
        ----------
        loss_list: List of float.
            The loss values at each epoch. 
        """
        
        print("Begin training")
        n_inst = X_train.shape[0]
        self.is_first_epoch = True
        loss_list = []
        for epoch_idx in range(1, n_epoch+1):
            # Zero out the gradient first
            self.zero_grad()
                           
            # Train
            loss = 0
            random_idx_list = [idx for idx in range(n_inst)]
            random.shuffle(random_idx_list)            
            for inst_idx in range(n_inst):
                # x_inst = X_train[inst_idx, :]
                # y_inst = y_train[inst_idx]
                
                x_inst = X_train[random_idx_list[inst_idx], :]
                y_inst = y_train[random_idx_list[inst_idx]]
                
                if np.size(y_inst) == 1:
                    y_inst = y_inst[:, np.newaxis]                
                # set_trace()
                
                # Forward pass
                # out = self.eval(x_inst)
                out, out_list = self.eval(x_inst)

                # Calculate the loss
                loss += self.loss(y_inst, out)
                
                # Backward pass
                self.backprop(y_inst)
                
                # Now update the gradient
                self.update_gradient(y_inst)
                                             
            # Take average of the gradient
            for layer_idx, layer in enumerate(self.layer_list):
                layer["weight_grad"] /= (1.0*n_inst)
                layer["bias_grad"] /= (1.0*n_inst)
            
            loss /= (1.0*n_inst)
            print("Epoch %d, loss value: %f" % (epoch_idx, loss))
            # set_trace()
            
            # Update the gradient based on the optimization algorithm
            if self.optimizer_name == "conjugate_gradient":
                self.optimizer.step(epoch_idx, self, X_train, y_train) # We need X_train and y_train for interval search 
            elif self.optimizer_name == "Adam":
                self.optimizer.step(epoch_idx-1, self)  # Adam algorithm starts from 0
            else:
                self.optimizer.step(epoch_idx, self) 
            
            # Add loss to list
            loss_list.append(loss)
            
            if self.is_first_epoch is True:
                self.is_first_epoch = False
           
        print("Finish training")
        
        return loss_list

def main():
    pass
    # test_gradient_descent()
    # test_gradient_descent_with_momentum()
    # set_trace()
    
    # set_trace()

if __name__ == "__main__":
    main()
