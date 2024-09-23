import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
import random

from utils import eval_activation_func, eval_activation_func_gradient

import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)
random.seed(72)

class NeuralNetwork:
    def __init__(self):
        pass

class Perceptron(NeuralNetwork):
    """ Perceptron. Here for the Widrow-Hoff experiment we evaluate the loss
    after updating the gradient at every step.
    
    Attributes
    ------------
    in_dim: Integer. 
        The input dimension. 
    out_dim: Integer. 
        The output dimension. 
    activation_func: String. 
        The name of the activation function. 
    learning_rate: Float. 
        The learning rate. Default is 0.03.
    weight: Numpy array. 
        The weight, of size (out_dim, in_dim). 
    bias: Numpy array.  
        The bias, of size (out_dim).  
    weight_grad: Numpy array. 
        The weight, of size (out_dim, in_dim). 
    bias_grad: Numpy array.  
        The bias, of size (out_dim).  
    leaky_alpha: Float. 
        The coefficient if LeakyRelu or ELU is used. Default is 0.1.
   
    Methods
    -----------
    eval(x_in)
        Evaluate the result given the input x_in.
    train(X_train, y_train, n_epoch)
        Train the network
    """
    
    def __init__(self, in_dim, out_dim, activation_func="linear", learning_rate=0.03, leaky_alpha=0.1):
        """
        Parameters
        ----------
        in_dim : Integer. 
            The input dimension. 
        out_dim : Integer. 
            The output dimension. 
        activation_func : String. 
            The name of the activation function. 
        learning_rate: Float. 
            The learning rate. Default is 0.03. 
        leaky_alpha: Float. 
            The coefficient if LeakyRelu or ELU is used. Default is 0.1.        
        """
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        self.leaky_alpha = leaky_alpha
        
        # self.weight = np.random.normal(size=(out_dim, in_dim))
        # self.bias = np.random.normal(size=(out_dim))
        
        self.weight = np.zeros((out_dim, in_dim))
        self.bias = np.zeros((out_dim))
      
        self.weight_grad = np.zeros((out_dim, in_dim))
        self.bias_grad = np.zeros((out_dim))
        
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
        
        out_mat_1 = np.dot(self.weight, x_in)
        out_mat = out_mat_1.squeeze() + self.bias.squeeze()
        if (self.activation_func == "LeakyRelu") or (self.activation_func == "elu"):
            out = eval_activation_func(out_mat, self.activation_func, leaky_alpha=self.leaky_alpha)
        else:
            out = eval_activation_func(out_mat, self.activation_func)
        return out
                
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
        loss_list = []
        for epoch_idx in range(1, n_epoch+1):
            # Train
            loss = 0
            random_idx_list = [idx for idx in range(n_inst)]
            random.shuffle(random_idx_list)            
            for inst_idx in range(n_inst):
                x_inst = X_train[random_idx_list[inst_idx], :]
                y_inst = y_train[random_idx_list[inst_idx]]
                
                if np.size(y_inst) == 1:
                    y_inst = y_inst[:, np.newaxis]                
               
                # Forward pass
                out = self.eval(x_inst)
                err = y_inst - out
                err = np.squeeze(err)
                
                # Update
                if (self.activation_func == "LeakyRelu") or (self.activation_func == "elu"):
                    self.weight += 2*self.learning_rate*err*eval_activation_func_gradient(out, self.activation_func, leaky_alpha=self.leaky_alpha)*x_inst
                    self.bias += 2*self.learning_rate*err*eval_activation_func_gradient(out, self.activation_func, leaky_alpha=self.leaky_alpha)
                else:
                    self.weight += 2*self.learning_rate*err*eval_activation_func_gradient(out, self.activation_func)*x_inst
                    self.bias += 2*self.learning_rate*err*eval_activation_func_gradient(out, self.activation_func)
                
                # Calculate sum of square loss
                loss = 0
                for idx2 in range(n_inst):
                    x_inst2 = X_train[idx2, :]
                    y_inst2 = y_train[idx2]
                    if np.size(y_inst2) == 1:
                        y_inst2 = y_inst2[:, np.newaxis]        
                    out2 = self.eval(x_inst2)
                    err2 = y_inst2 - out2
                    err2 = np.squeeze(err2)
                    loss += np.sum(err2**2) # MSE                
                loss_list.append(loss)
                
        return loss_list

def get_experiment_data():
    """ Get the experimental data of the Widrow-Hoff experiment, based on descriptions in [1] (exercise P10.9).
        
    [1] M. Hagan et al., Neural network design (2nd ed.), 2014, .        

    Parameters
    ---------------
    None.
    
    Returns
    ----------
    X_train: Numpy array. 
        The training data, of size (n_inst, n_features) where
        n_inst is the number of instances, and n_features is 
        the number of features.
    y_train: Numpy array.
        The ground truth, of size (n_inst).
    """
        
    X_train = np.array([
        [1, -1, -1, -1,    1, 1, 1, 1,    1, -1, -1, -1,    -1, -1, -1, -1   ],
        [1, 1, 1, 1,    1, -1, 1, 1,    1, -1, 1, 1,    -1, -1, -1, -1],
        [1, 1, 1, 1,    1, 1, -1, -1,    1, -1, -1, -1,    -1, -1, -1, -1],
        [-1, -1, -1, -1,    1, -1, -1, -1,    1, 1, 1, 1,    1, -1, -1, -1],
        [-1, -1, -1, -1,    1, 1, 1, 1,    1, -1, 1, 1,    1, -1, 1, 1],        
        [-1, -1, -1, -1,    1, 1, 1, 1,    1, 1, -1, -1,    1, -1, -1, -1],
    ])
    y_train = np.array([60, 0, -60, 60, 0, -60])
    y_train = y_train[:, np.newaxis]
    return (X_train, y_train)

def perform_widrow_hoff_experiment():
    """ Perform the Widrow-Hoff experiment, based on descriptions in [1] (exercise P10.9).
        
    [1] M. Hagan et al., Neural network design (2nd ed.), 2014, .        

    Parameters
    ---------------
    None.
    
    Returns
    ----------
    X_train: Numpy array. 
        The training data, of size (n_inst, n_features) where
        n_inst is the number of instances, and n_features is 
        the number of features.
    y_train: Numpy array.
        The ground truth, of size (n_inst).
    """
    
    (X_train, y_train) = get_experiment_data()
    (n_inst, n_features) = X_train.shape

    net = Perceptron(n_features, 1, "linear")
    # net = Perceptron(n_features, 1, "linear", learning_rate=0.09)
    n_epoch = 20
    loss_list = net.train(X_train, y_train, n_epoch)
    loss_list = loss_list[:100]

    plt.plot(range(1, len(loss_list)+1), loss_list, color="blue", label="Sum of square error")
    plt.legend()
    plt.ylim((0, 2.5e+04))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Time step")
    plt.title("Widrow-Hoff experiment")
    plt.show()

def perform_widrow_hoff_experiment_leaky_relu():
    """ Perform the Widrow-Hoff experiment, based on descriptions in [1] (exercise P10.9).
    But this time we use Leaky ReLU (alpha parameter set to 0.1, I tried 0.01 but it did not work).
        
    [1] M. Hagan et al., Neural network design (2nd ed.), 2014, .        

    Parameters
    ---------------
    None.
    
    Returns
    ----------
    X_train: Numpy array. 
        The training data, of size (n_inst, n_features) where
        n_inst is the number of instances, and n_features is 
        the number of features.
    y_train: Numpy array.
        The ground truth, of size (n_inst).
    """
    
    (X_train, y_train) = get_experiment_data()
    (n_inst, n_features) = X_train.shape

    # net = Perceptron(n_features, 1, "linear")
    net = Perceptron(n_features, 1, "LeakyRelu", learning_rate=0.03, leaky_alpha=0.1)
    n_epoch = 2000
    loss_list = net.train(X_train, y_train, n_epoch)

    plt.plot(range(1, len(loss_list)+1), loss_list, color="blue", label="Sum of square error")
    plt.legend()
    # plt.ylim((0, 2.5e+04))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Time step")
    plt.title("Widrow-Hoff experiment (Leaky ReLU)")
    plt.show()

def perform_widrow_hoff_experiment_elu():
    """ Perform the Widrow-Hoff experiment, based on descriptions in [1] (exercise P10.9).
    But this time we use ELU.
        
    [1] M. Hagan et al., Neural network design (2nd ed.), 2014, .        

    Parameters
    ---------------
    None.
    
    Returns
    ----------
    X_train: Numpy array. 
        The training data, of size (n_inst, n_features) where
        n_inst is the number of instances, and n_features is 
        the number of features.
    y_train: Numpy array.
        The ground truth, of size (n_inst).
    """
    
    (X_train, y_train) = get_experiment_data()
    (n_inst, n_features) = X_train.shape

    # learning_rate_list = np.arange(0.001, 1.0, 0.001)
    
    # learning_rate_list = np.arange(0.032, 1.0, 0.001)
    # leaky_alpha_list = np.arange(0.001, 1.0, 0.001)
    
    # learning_rate_list = np.arange(0.01, 1.0, 0.01)
    # leaky_alpha_list = np.arange(0.01, 1.0, 0.01)
    
    # for learning_rate in learning_rate_list:
        # q1 = 0
        # for leaky_alpha in leaky_alpha_list:            
            # print("Learning rate: %f, leaky alpha: %f" % (learning_rate, leaky_alpha))
            # net = Perceptron(n_features, 1, "elu", learning_rate=learning_rate, leaky_alpha=leaky_alpha)
            # n_epoch = 1000
            # loss_list = net.train(X_train, y_train, n_epoch)
            # loss = loss_list[-1]
            # print("Loss: %f" % (loss))
            # if loss < 0.005:
                # print("Found:")
                # print(learning_rate)
                # print(leaky_alpha)
                # q1 = 1
                # break
        # if q1 == 1:
            # break
    # set_trace()
    
    net = Perceptron(n_features, 1, "elu", learning_rate=0.06, leaky_alpha=0.1)
    n_epoch = 1000
    loss_list = net.train(X_train, y_train, n_epoch)
    # loss_list = loss_list[-1000:]
    
    plt.plot(range(1, len(loss_list)+1), loss_list, color="blue", label="Sum of square error")
    plt.legend()
    # plt.ylim((0, 2.5e+04))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Time step")
    plt.title("Widrow-Hoff experiment (ELU)")
    plt.show()

def main():    
    perform_widrow_hoff_experiment()
    perform_widrow_hoff_experiment_leaky_relu()
    # perform_widrow_hoff_experiment_elu()
    
if __name__ == "__main__":
    main()
