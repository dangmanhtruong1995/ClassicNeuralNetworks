import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
import random

from utils import eval_activation_func, eval_activation_func_gradient
# from widrow_hoff_experiment import Perceptron
from config import get_config

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=get_config()["debug_mode"])

class Perceptron:
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
                if isinstance(y_inst, int) is False:
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


class RadialBasisNetwork:
    def __init__(self):
        self.delta = 0.001 # Threshold parameter
        
    def train(self, X_train, y_train, debug=True):        
        (n_inst, n_features) = X_train.shape
        n_neuron_layer1 = n_inst
        self.w_layer1 = X_train.copy()
        self.b_layer1 = np.ones((n_neuron_layer1, 1))
      
        # Get output of the first layer w.r.t all training instances
        self.U_mat = np.ones((n_inst+1, n_inst))
        for inst_idx in range(n_inst):
            inst = X_train[inst_idx, :]
            act = np.zeros((n_neuron_layer1))
            diff = inst-self.w_layer1
            dist = np.linalg.norm(diff, axis=1)
            if len(dist.shape) == 1:
                dist = dist[:, np.newaxis]
            dist = np.multiply(dist, self.b_layer1)
            act = np.exp(-dist**2)
            if np.size(act) > 1:
                act = np.squeeze(act)
            logging.debug("Instance %d, activation for layer 1:" % (inst_idx+1))
            logging.debug(act)
            self.U_mat[:-1,  inst_idx] = act # The U matrix also stores the bias, which are set to 1
            
        logging.debug(self.U_mat)
        
        self.U_mat = np.transpose(self.U_mat)
        
        # Use the Orthogonal Least Square (OLS) algorithm for finding the optimal set of chosen subsets in the first layer
        # and the optimal weights in the second layer
        step_idx = 0
        chosen_subset_list = []
        o_val_chosen_list = [] 
        m_vect_opt_list = []
        # r_mat_list = []
        r_mat = {}
        r_mat_opt = {}
        h_mat = np.ones((n_neuron_layer1+1, n_inst+1))*(-999999)
        h_opt_list = []
        while True:
            o_val_list = np.zeros((n_neuron_layer1+1))
            m_vect_list = []
            h_val_list = []
            for neuron_layer1_idx in range(n_neuron_layer1+1): # Includes bias
                if neuron_layer1_idx in chosen_subset_list:
                    o_val_list[neuron_layer1_idx] = -99999
                    m_vect_list.append(None)
                    h_val_list.append(-99999)
                    continue

                logging.debug("Step %d, index %d" % (step_idx+1, neuron_layer1_idx))
                
                # Calculate contribution of each potential basis
                if step_idx == 0:
                    m_vect = self.U_mat[:, neuron_layer1_idx].copy()
                    if len(m_vect.shape) == 1:
                        m_vect = m_vect[:, np.newaxis]
                    
                    numerator = np.matmul(np.transpose(m_vect), y_train)
                    denominator = 1.0*(np.matmul(np.transpose(m_vect), m_vect)[0][0])                
                    h_val = numerator/denominator                
                    o_val = ((h_val**2)*denominator)/(1.0*(np.matmul(np.transpose(y_train), y_train)))                
                    
                    logging.debug("h_val:")
                    logging.debug(h_val)
                    logging.debug("o_val:")
                    logging.debug(o_val)
                    
                    o_val_list[neuron_layer1_idx] = o_val
                    m_vect_list.append(m_vect)
                    h_mat[0, neuron_layer1_idx] = h_val
                    if np.size(h_val) == 1:
                        h_val = h_val.item()
                    h_val_list.append(h_val)
                else:
                    for prev_step_idx in range(step_idx):
                        m_vect_prev = m_vect_opt_list[prev_step_idx]     

                        u_vect = self.U_mat[:, neuron_layer1_idx].copy()
                        if len(u_vect.shape) == 1:
                            u_vect = u_vect[:, np.newaxis]

                        numerator_r = np.matmul(np.transpose(m_vect_prev), u_vect)
                        denominator_r = 1.0*(np.matmul(np.transpose(m_vect_prev), m_vect_prev)[0][0])    
                        r_mat[(neuron_layer1_idx, prev_step_idx, step_idx)] = numerator_r / denominator_r
                        if np.size(r_mat[(neuron_layer1_idx, prev_step_idx, step_idx)]) == 1:
                            r_mat[(neuron_layer1_idx, prev_step_idx, step_idx)] = r_mat[(neuron_layer1_idx, prev_step_idx, step_idx)].item()
                        logging.debug("r_mat[(neuron_layer1_idx, prev_step_idx, step_idx)]")
                        logging.debug(r_mat[(neuron_layer1_idx, prev_step_idx, step_idx)])
                    
                    m_vect = self.U_mat[:, neuron_layer1_idx].copy()
                    if len(m_vect.shape) == 1:
                        m_vect = m_vect[:, np.newaxis]
                    for prev_step_idx in range(step_idx):
                        m_vect_prev = m_vect_opt_list[prev_step_idx]
                        m_vect -= (r_mat[(neuron_layer1_idx, prev_step_idx, step_idx)]*m_vect_prev)
                    logging.debug("m_vect in gram-schmidt")
                    logging.debug(m_vect)
                    
                    numerator = np.matmul(np.transpose(m_vect), y_train)[0][0]
                    denominator = 1.0*(np.matmul(np.transpose(m_vect), m_vect)[0][0])                
                    h_val = numerator/denominator                
                    o_val = ((h_val**2)*denominator)/(1.0*(np.matmul(np.transpose(y_train), y_train)))
                    
                    if o_val == 0:
                        logging.debug("IS NAN!!!!")
                        set_trace()
                    
                    logging.debug("h_val:")
                    logging.debug(h_val)
                    logging.debug("o_val:")
                    logging.debug(o_val)            

                    o_val_list[neuron_layer1_idx] = o_val
                    m_vect_list.append(m_vect)
                    h_mat[step_idx, neuron_layer1_idx] = h_val
                    h_val_list.append(h_val)
 
            chosen_subset = np.argmax(o_val_list[:-1]) # Don't take the bias part
            h_val_opt = h_val_list[chosen_subset]
           
            try:
                m_vect_opt = m_vect_list[chosen_subset]
            except:
                print("BIG ERROR!")
                set_trace()
            
            if step_idx > 0:
                for prev_step_idx in range(step_idx):
                    r_mat_opt[(prev_step_idx, step_idx)] = r_mat[(chosen_subset, prev_step_idx, step_idx)]
            
            o_val_chosen_list.append(o_val_list[chosen_subset])
            chosen_subset_list.append(chosen_subset)
            m_vect_opt_list.append(m_vect_opt)
            h_opt_list.append(h_val_opt)
            
            logging.debug("Step %d, chosen center: %d, error reduced: %f" % (step_idx+1, chosen_subset+1, o_val_list[chosen_subset]))
            logging.debug(chosen_subset_list)
            
            step_idx += 1
            
            if debug is False:
                if (1 - np.sum(np.array(o_val_chosen_list))) < self.delta:
                    print("Error smaller than threshold. Exiting")
                    break
            
            if len(chosen_subset_list) == n_inst:
                print("All instances have been chosen. Breaking while loop")
                break
        
        logging.debug("Chosen subset list")
        logging.debug(chosen_subset_list)
        
        m_vect_last = np.ones((n_inst))
        numerator = np.matmul(np.transpose(m_vect), y_train)[0][0]
        denominator = 1.0*(np.matmul(np.transpose(m_vect), m_vect)[0][0])          
        h_last = numerator / denominator
        o_val_last = ((h_val**2)*denominator)/(1.0*(np.matmul(np.transpose(y_train), y_train)))
        n_chosen = len(chosen_subset_list)
        h_opt_list.append(h_last)
        
        self.chosen_subset_list = chosen_subset_list
        # set_trace()
        
        # Use the result of OLS to initialize the first layer
        self.chosen_subset_list = self.chosen_subset_list[:5]        
        n_subset = len(self.chosen_subset_list)        
        n_neuron_layer1 = n_subset
        self.w_layer1 = np.zeros((n_neuron_layer1, n_features))
        self.b_layer1 = np.ones((n_neuron_layer1, 1))
        for subset_idx in range(n_subset):
            self.w_layer1[subset_idx, :] = X_train[self.chosen_subset_list[subset_idx], :]
        
        # Get output of the first layer
        act = np.zeros((n_neuron_layer1, n_inst))
        for inst_idx in range(n_inst):
            for neuron_layer1_idx  in range(n_neuron_layer1):
                diff = self.w_layer1[neuron_layer1_idx, :]-np.squeeze(X_train[inst_idx, :])
                # set_trace()
                dist = np.linalg.norm(diff, axis=0)
                if len(dist.shape) == 1:
                    dist = dist[:, np.newaxis]
                dist = np.multiply(dist, self.b_layer1[neuron_layer1_idx])
                act1 = np.exp(-dist**2)
                if np.size(act1) > 1:
                    act1 = np.squeeze(act1)
                act[neuron_layer1_idx, inst_idx] = act1        
        act = np.transpose(act)
        
        # Train the second layer using the LMS algorithm
        logging.debug("Train the second layer using the LMS algorithm")
        net = Perceptron(n_neuron_layer1, 1)
        n_epoch = 100    
        y_train1 = np.squeeze(y_train)
        net.train(act, y_train, n_epoch)
        self.w_layer2 = net.weight
        self.b_layer2 = net.bias
        
        logging.debug("Finished training the second layer using the LMS algorithm")
        # set_trace()
      
    def eval(self, X_eval):
        logging.debug("Eval")
        # Use the result of OLS to initialize the first layer
        (n_inst, n_features) = X_eval.shape
        # chosen_subset_list = self.chosen_subset_list[:3]
        chosen_subset_list = self.chosen_subset_list
        n_subset = len(chosen_subset_list)        
        n_neuron_layer1 = n_subset
                
        # Get output of the first layer
        act = np.zeros((n_neuron_layer1, n_inst))
        for inst_idx in range(n_inst):
            for neuron_layer1_idx  in range(n_neuron_layer1):
                diff = self.w_layer1[neuron_layer1_idx, :]-np.squeeze(X_eval[inst_idx, :])
                # set_trace()
                dist = np.linalg.norm(diff, axis=0)
                if len(dist.shape) == 1:
                    dist = dist[:, np.newaxis]
                dist = np.multiply(dist, self.b_layer1[neuron_layer1_idx])
                act1 = np.exp(-dist**2)
                if np.size(act1) > 1:
                    act1 = np.squeeze(act1)
                act[neuron_layer1_idx, inst_idx] = act1        
        act = np.transpose(act)
        # set_trace()
        
        # Get output of the second layer
        out_mat_1 = np.dot(self.w_layer2, np.transpose(act))
        out_mat = out_mat_1.squeeze() + self.b_layer2.squeeze()
        
        return out_mat

def create_data(): 
    X_train = np.arange(-1, 1.2, 0.5)
    X_train = X_train[:, np.newaxis]
    y_train = np.zeros(X_train.shape[0])
    for inst_idx in range(X_train.shape[0]):
        inst = X_train[inst_idx, :]
        y_train[inst_idx] = np.cos(np.pi*inst)
    y_train = y_train[:, np.newaxis]    
    # y_train[1] = 0
    # y_train[3] = 0
    # y_train += 0.00001
    return (X_train, y_train)

   
def main():    
    (X_train, y_train) = create_data()
    
    rbf = RadialBasisNetwork()
    rbf.train(X_train, y_train, debug=True)
    y_pred = rbf.eval(X_train)
    
    print(y_pred)
    print(y_train)
    # Output
    # [-0.98091243  0.09578187  0.71150487  0.09672919 -0.98106787]
    # [[-1.000000e+00]
     # [ 6.123234e-17]
     # [ 1.000000e+00]
     # [ 6.123234e-17]
     # [-1.000000e+00]]

if __name__ == "__main__":
    main()    

