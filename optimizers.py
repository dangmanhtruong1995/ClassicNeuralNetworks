import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
import logging
import copy
import random

from utils import get_total_number_of_parameters, flatten_into_vector, unflatten_from_vector
from config import get_config

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=get_config()["debug_mode"])


def evaluate_at_weight_pos(net, w_curr, X_train, y_train):
    """ Run a forward pass of the network on the training dataset and calculate the loss using a specified weight. 
        
    Parameters
    ---------------
    net : a NeuralNetwork object.
        The network whose weights will be changed to w_curr and then
        a forward pass will be run to calculate the loss.      
    w_curr : Numpy array.
        The array of the total weights of the networks, as a one-dimensional
        array. It stores in the format [layer-1-weight, layer-1-bias, layer-2-weight, 
        layer-2-bias, ..., layer-N-weight, layer-N-bias] where N is the total number
        of layers, and the weight and bias in the each layer are flattened as well. 
    X_train: Numpy array. 
        The training data, of size (n_inst, n_features) where
        n_inst is the number of instances, and n_features is 
        the number of features.
    y_train: Numpy array.
        The ground truth, of size (n_inst).
    
    Returns
    -------
    loss: Float.
        The loss value calculated during the forward pass. 
    """
    
    unflatten_from_vector(net, w_curr, is_grad=False)
    # loss = perform_forward_pass(net, X_train, y_train)
    loss = 0 
    n_inst = X_train.shape[0]
    random_idx_list = [idx for idx in range(n_inst)]
    random.shuffle(random_idx_list)    
    for inst_idx in range(n_inst):
        x_inst = X_train[random_idx_list[inst_idx], :]
        y_inst = y_train[random_idx_list[inst_idx]]
        
        # Forward pass
        out, _ = net.eval(x_inst)

        # Calculate the loss
        loss += net.loss(y_inst, out)
        
    loss /= (1.0*n_inst)
    
    return loss

class GradientDescentOptimizer:
    """ Gradient descent optimizer.
    
    Attributes
    ------------
    learning_rate : Float 
        The learning rate. 

    Methods
    -----------
    step(epoch_idx, net)
        Run one epoch of the gradient descent algorithm.     
    """
    
    def __init__(self, optimizer_params):
        """
        Parameters
        ----------
        optimizer_params : Dictionary.
            The parameters of the optimization algorithm.
        """

        self.learning_rate = optimizer_params["learning_rate"]
        
    def step(self, epoch_idx, net):
        """ Run one epoch of the gradient descent algorithm.   

        Parameters
        ---------------
        epoch_idx : Integer.
            The current epoch number.
        net : a NeuralNetwork object.
            The network whose weights will be updated.            

        Returns
        -------
        None. 
        """
        
        for layer_idx, layer in enumerate(net.layer_list):
            layer["weight"] -= self.learning_rate*layer["weight_grad"]
            layer["bias"] -= self.learning_rate*layer["bias_grad"]

class AdamOptimizer:
    """ Adam optimizer, described in [1].
    
    [1] - D. P. Kingma et al., Adam: A method for stochastic optimization, ICLR, 2015. 
    
    Attributes
    ------------
    learning_rate : Float.
        The learning rate (default is 0.001). 
    beta1: Float.
        The exponential decay rate for the first moment (default is 0.9).
    beta2: Float.
        The exponential decay rate for the second moment (default is 0.999).
    epsilon: Float. 
        Very small number to prevent divison overflow.

    Methods
    -----------
    step(epoch_idx, net)
        Run one epoch of the gradient descent algorithm.     
    """
    
    def __init__(self, optimizer_params):
        """
        Parameters
        ----------
        optimizer_params : Dictionary.
            The parameters of the optimization algorithm.
        """
        if "learning_rate" in optimizer_params:
            self.learning_rate = optimizer_params["learning_rate"]
        else:
            self.learning_rate = 0.001
            
        if "beta1" in optimizer_params:
            self.beta1 = optimizer_params["beta1"]
        else:
            self.beta1 = 0.9
            
        if "beta2" in optimizer_params:
            self.beta2 = optimizer_params["beta2"]
        else:
            self.beta2 = 0.999
        
        self.epsilon = 10**(-8)
        
    def step(self, epoch_idx, net):
        """ Run one epoch of the gradient descent algorithm.   

        Parameters
        ---------------
        epoch_idx : Integer.
            The current epoch number.
        net : a NeuralNetwork object.
            The network whose weights will be updated.            

        Returns
        -------
        None. 
        """ 
        
        if epoch_idx == 0:
            # Initialize 
            lr_curr = self.learning_rate
            first_moment_list = []
            second_moment_list = []
            for layer_idx, layer in enumerate(net.layer_list):
                first_moment = {
                    "weight_mm": np.zeros_like(layer["weight_grad"]), 
                    "bias_mm": np.zeros_like(layer["bias_grad"]),                     
                }
                second_moment = {
                    "weight_mm": np.zeros_like(layer["weight_grad"]), 
                    "bias_mm": np.zeros_like(layer["bias_grad"]),                     
                }
                first_moment_list.append(first_moment)
                second_moment_list.append(second_moment)
        else:
            first_moment_list = self.first_moment_list
            second_moment_list  = self.second_moment_list        
        
        # Update first and second moment estimates
        for layer_idx, layer in enumerate(net.layer_list):
            first_moment_list[layer_idx]["weight_mm"] = self.beta1*first_moment_list[layer_idx]["weight_mm"] + (1-self.beta1)*layer["weight_grad"]
            first_moment_list[layer_idx]["bias_mm"] = self.beta1*first_moment_list[layer_idx]["bias_mm"] + (1-self.beta1)*layer["bias_grad"]
            
            second_moment_list[layer_idx]["weight_mm"] = self.beta2*second_moment_list[layer_idx]["weight_mm"] + (1-self.beta2)*np.square(layer["weight_grad"])
            second_moment_list[layer_idx]["bias_mm"] = self.beta2*second_moment_list[layer_idx]["bias_mm"] + (1-self.beta2)*np.square(layer["bias_grad"])
       
        # Bias correction
        if epoch_idx > 0:
            lr_curr = self.learning_rate*(np.sqrt(1-np.power(self.beta2, epoch_idx))/(1-np.power(self.beta1, epoch_idx)))
        
        # Update the network parameters
        for layer_idx, layer in enumerate(net.layer_list):
            layer["weight"] -= (lr_curr*first_moment_list[layer_idx]["weight_mm"]/(np.sqrt(second_moment_list[layer_idx]["weight_mm"])+self.epsilon))
            layer["bias"] -= (lr_curr*first_moment_list[layer_idx]["bias_mm"]/(np.sqrt(second_moment_list[layer_idx]["bias_mm"])+self.epsilon))

        self.first_moment_list = first_moment_list
        self.second_moment_list = second_moment_list

class ConjugateGradientOptimizer:
    """ Conjugate gradient optimizer.
    
    Attributes
    ----------
    method : String. 
        The method used for finding the beta coefficient. Supported methods are:
            - fletcher_and_reeves
            - hestenes_and_stiefel
            - polak_and_ribiere
    n_dim: Integer.    
        The total number of parameters in the network. Used for restarting the search procedure periodically.
    grad_prev: Numpy array.
        The gradient of the previous epoch.
    p_prev: Numpy array.
        The search direction of the previous epoch.
    grad_prev_mult_grad_prev: Numpy array.
        The gradient of the previous epoch multiplied by itself. Used when method is 'fletcher_and_reeves'.
     
    Methods
    -------
    step(epoch_idx, net, X_train, y_train)
        Run one epoch of the gradient descent algorithm.     
    """
    
    def __init__(self, optimizer_params):
        """
        Parameters
        ----------
        optimizer_params : Dictionary.
            The parameters of the optimization algorithm.
        """
        
        self.method = optimizer_params["method"]
        self.n_dim = None

        self.grad_prev = None
        self.p_prev = None
        self.grad_prev_mult_grad_prev = None # For the fletcher_and_reeves method
        
        # self.w_curr = None
  
    def step(self, epoch_idx, net, X_train, y_train):
        """ Run one epoch of the gradient descent algorithm.   

        Parameters
        ---------------
        epoch_idx : Integer.
            The current epoch number.
        net : a NeuralNetwork object.
            The network whose weights will be updated.            
        X_train: Numpy array. 
            The training data, of size (n_inst, n_features) where
            n_inst is the number of instances, and n_features is 
            the number of features.
        y_train: Numpy array.
            The ground truth, of size (n_inst).
        
        Returns
        -------
        None. 
        """
        
        grad = flatten_into_vector(net, is_grad=True)

        w_curr = flatten_into_vector(net, is_grad=False)
        if self.n_dim is None:
            self.n_dim = get_total_number_of_parameters(net)
            
        net_copy = copy.deepcopy(net) # For safety reason
        
        if (epoch_idx == 1) or ((epoch_idx-1) % self.n_dim == 0):
            # First step is the same as with steepest descent
            # After every n_dim step the conjugate gradient algorithm is repeated
            p_curr = -grad            
            self.grad_prev = None
            self.p_prev = None
            self.grad_prev_mult_grad_prev = None # For the fletcher_and_reeves method
            pass
        else:
            if self.method == "fletcher_and_reeves":
                grad_mult_grad = np.dot(grad.transpose(), grad)
                # denominator = np.dot(grad_prev.transpose(), grad_prev)
                if self.grad_prev_mult_grad_prev is None:
                    self.grad_prev_mult_grad_prev = np.dot(self.grad_prev.transpose(), self.grad_prev)
                # beta = grad_mult_grad / (self.grad_prev_mult_grad_prev) 
                beta = grad_mult_grad / (self.grad_prev_mult_grad_prev + 1e-6) # For numerical stability                
                logging.debug("Beta: %f" % (beta))
                p_curr = -grad + beta*self.p_prev  
                # set_trace()
                self.grad_prev_mult_grad_prev = copy.deepcopy(grad_mult_grad) # For more efficient calculation
            elif self.method == "hestenes_and_stiefel":
                diff_grad_prev = grad - self.grad_prev
                diff_grad_prev_mult_grad = np.dot(diff_grad_prev, grad)
                diff_grad_prev_mult_p_prev = np.dot(diff_grad_prev, self.p_prev)                
                beta = diff_grad_prev_mult_grad / diff_grad_prev_mult_p_prev
                logging.debug("Beta: %f" % (beta))
                p_curr = -grad + beta*self.p_prev 
                # set_trace()
            elif self.method == "polak_and_ribiere":
                diff_grad_prev = grad - self.grad_prev
                diff_grad_prev_mult_grad = np.dot(diff_grad_prev, grad)
                grad_prev_mult_grad_prev = np.dot(self.grad_prev.transpose(), self.grad_prev)
                beta = diff_grad_prev_mult_grad / grad_prev_mult_grad_prev               
                logging.debug("Beta: %f" % (beta))
                p_curr = -grad + beta*self.p_prev 
            else:
                raise Exception("In ConjugateGradientOptimizer, method 'step': CG algorithm %s not supported" % (self.method))
      
        # Calculate learning rate using interval reduction and golden section search 
        interval_func = lambda alpha: evaluate_at_weight_pos(net_copy, w_curr+alpha*p_curr, X_train, y_train)
        # alpha_curr = ConjugateGradientOptimizer._perform_interval_search(interval_func)  
        alpha_curr = ConjugateGradientOptimizer._perform_interval_search(interval_func, tol=0.001)         
        
        # Step
        w_new = w_curr + alpha_curr*p_curr
        unflatten_from_vector(net, w_new, is_grad=False)
        
        # Prepare for the next epoch
        self.p_prev = copy.deepcopy(p_curr)
        self.grad_prev = copy.deepcopy(grad)
        # self.w_curr = copy.deepcopy(w_new)
        
    @staticmethod
    def _perform_interval_reduction(func, eps):
        """ Perform interval reduction procedure, based on descriptions in [1] (chapter 9 and 12).
        
        [1] M. Hagan et al., Neural network design (2nd ed.), 2014.        

        Parameters
        ---------------
        func : One-variable function.
            The function whose best interval range we want to find. 
        eps : Float.
            The initial step size.  
        
        Returns
        -------
        (left_interval, right_interval): Tuple of 2 float values.
            The reduced interval. 
        """
        
        logging.debug("Perform interval reduction")
    
        step = 0
        F_prev_2 = func(step*eps)
        logging.debug("Step: %d" % (step))
        logging.debug("Function value: %f" % (F_prev_2))
        
        step = 1    
        F_prev_1 = func(step*eps)
        logging.debug("Step: %d" % (step))
        logging.debug("Function value: %f" % (F_prev_1))   
        
        step *= 2
        F_curr = func(step*eps)
        logging.debug("Step: %d" % (step))
        logging.debug("Function value: %f" % (F_curr))
        
        if F_curr > F_prev_1:
            left_interval = (step / 4)*eps
            right_interval = step*eps
            return (left_interval, right_interval)
        
        while True:
            step *= 2
            F_new = func(step*eps)
            logging.debug("Step (interval reduction): %d" % (step))
            logging.debug("Function value: %f" % (F_new))
            
            if F_new > F_curr:
                left_interval = (step / 4)*eps
                right_interval = step*eps
                return (left_interval, right_interval)
            
            F_prev_2 = F_prev_1
            F_prev_1 = F_curr
            F_curr = F_new  

    @staticmethod
    def _perform_golden_section_search(func, left_interval, right_interval, tau, tol):
        """ Perform golden section search, based on descriptions in [1] (chapter 9 and 12).
        
        [1] M. Hagan et al., Neural network design (2nd ed.), 2014.        

        Parameters
        ---------------
        func : One-variable function.
            The function whose best value. 
        left_interval : Float.
            The left side of the interval found by the interval reduction procedure.  
        right_interval : Float.
            The right side of the interval found by the interval reduction procedure.
        tau: Float.
            The search ratio in each step.
        tol: Float.
            The accuracy tolerance set by the user. 
        
        Returns
        -------
        opt_val: Float.
            The optimal value. 
        """
        
        a_curr = left_interval
        b_curr = right_interval
        
        c_curr = a_curr + (1-tau)*(b_curr-a_curr)
        d_curr = b_curr - (1-tau)*(b_curr-a_curr)
        # set_trace()
        Fc = func(c_curr)
        Fd = func(d_curr)
        
        logging.debug("Begin Golden section search")
        logging.debug("a = %f, b=%f, c=%f, d=%f" % (a_curr, b_curr, c_curr, d_curr))    
        logging.debug("Fc = %f" % (Fc))
        logging.debug("Fd = %f" % (Fd))
        
        iteration= 0
        while True:
            iteration += 1
            logging.debug("Iteration (golden section search): %d" % (iteration)) 
            
            if Fc < Fd:
                b_curr = d_curr
                d_curr = c_curr            
                c_curr = a_curr + (1-tau)*(b_curr-a_curr)
                Fd = Fc
                Fc = func(c_curr)            
            else:
                a_curr = c_curr
                c_curr = d_curr
                d_curr = b_curr - (1-tau)*(b_curr-a_curr)
                Fc = Fd
                Fd = func(d_curr)
                
            logging.debug("a = %f, b=%f, c=%f, d=%f" % (a_curr, b_curr, c_curr, d_curr))    
            logging.debug("Fc = %f" % (Fc))
            logging.debug("Fd = %f" % (Fd))
            
            if (b_curr-a_curr) < tol:
                # logging.debug("b-a smaller than tolerance level. Terminating")
                break
                
        opt_val =  (b_curr+a_curr) / 2.0      
        logging.debug("Result: %f" % (opt_val))
        logging.debug("")
         
        return opt_val  

    @staticmethod
    def _perform_interval_search(func, eps=0.075, tau=0.618, tol=0.01):
        """ Perform interval search, based on descriptions in [1] (chapter 9 and 12).
        Internally, this function will call _perform_interval_reduction and _perform_golden_section_search.        
        
        [1] M. Hagan et al., Neural network design (2nd ed.), 2014.        

        Parameters
        ---------------
        func : One-variable function.
            The function whose best value.         
        eps : Float.
            The initial step size. Default is 0.075.
        tau: Float.
            The search ratio in each step. Default is 0.618 (based on the golden ratio).
        tol: Float.
            The accuracy tolerance set by the user. Default is 0.01
        
        Returns
        -------
        opt_val: Float.
            The optimal value. 
        """
        
        (left_interval, right_interval) = ConjugateGradientOptimizer._perform_interval_reduction(func, eps)
        logging.debug("Result by interval reduction:")
        logging.debug((left_interval, right_interval))

        alpha_opt = ConjugateGradientOptimizer._perform_golden_section_search(func, left_interval, right_interval, tau, tol)
        logging.debug("Learning rate found by golden section search: %f" % (alpha_opt))
        
        return alpha_opt
