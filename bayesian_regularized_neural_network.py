import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
import random
import time

# import numba
# from numba import njit
# from numba.typed import Dict, List
# from numba.types import float32, unicode_type, ListType

from utils import eval_activation_func, eval_activation_func_gradient, eval_loss_func, \
    flatten_into_vector, unflatten_from_vector, get_total_number_of_parameters, \
    log_time, show_info
from config import get_config
from neural_network_regression import GenericFeedforwardNeuralNetwork

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=get_config()["debug_mode"])
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_jacobian_matrix(X_train, n_inst, n_total_params, n_layer, layer_list, random_idx_list,
        out_list_all, m_ss_all, out_dim):
    jacobian = np.zeros((out_dim*n_inst, n_total_params))
    for inst_idx in range(n_inst):
        for out_dim_idx in range(out_dim):
            row_idx = inst_idx*out_dim + out_dim_idx
            col_idx = 0
            for layer_idx in range(0, n_layer):
                # Weight
                for i1_idx in range(layer_list[layer_idx]["out_curr_dim"]):
                    for j1_idx in range(layer_list[layer_idx]["in_curr_dim"]):
                        if layer_idx == 0:
                            a_m_minus_one = X_train[random_idx_list[inst_idx], j1_idx]
                        else:
                            a_m_minus_one = out_list_all[inst_idx][layer_idx-1][j1_idx]
                        jacobian[row_idx, col_idx] = m_ss_all[layer_idx][i1_idx, row_idx]*a_m_minus_one
                        col_idx += 1
                        
                # Bias
                for i1_idx in range(layer_list[layer_idx]["out_curr_dim"]):
                    jacobian[row_idx, col_idx] = m_ss_all[layer_idx][i1_idx, row_idx]
                    col_idx += 1
    return jacobian


class BayesianRegularizedNeuralNetwork(GenericFeedforwardNeuralNetwork):
    """ Neural network using Bayesian regularization for training.
    
    Attributes
    ------------
    in_dim: Integer. 
        The input dimension. 
    hidden_dim_list: List of integer.
        The list of number of hidden neurons in each hidden layer. Perceptron mode is 
        not supported so if this variable is an empty list, then an error will be raised.
    out_dim: Integer. 
        The output dimension. 
    activation_func_list: List of string. 
        The list of the name of the activation function in each layer. 
    mu: Float.
        The coefficient which will be multipled with the identity matrix in order to help with matrix inversion.
    mult_factor: Float.
        The multiplication factor to update mu. 
    gradnorm_thres: Float.
        The threshold of the total gradient norm below which training will be terminated.
    layer_list: List of dictionaries.
        A list of layers, with each layer being a dictionary containing the following keys:
            - 'weights': Storing the weight at the current layer
            - 'bias': Storing the bias at the current layer
    n_hidden_layer: Integer.
        Number of hidden layers.
    n_layer: Integer.
        Number of total layers.
    
    Methods
    -----------
    eval(x_in)
        Evaluate the result given the input x_in.
    train(X_train, y_train, n_epoch)
        Train the network.
    """

    def __init__(self, in_dim, hidden_dim_list, out_dim, 
            activation_func_list, optimizer_params, init_type="random", randomize=True):
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
            The dictionary storing the parameters. For this algorithm the followings are used:
                - mu: The coefficient which will be multipled with the identity matrix in order to help with matrix inversion.
                - mult_factor: The multiplication factor to update mu.
                - gradnorm_thres: The threshold of the total gradient norm below which training will be terminated.
        init_type: String.
            Initialization type. Default is 'random'. 
            - 'random': Weights are randomly initialized.
            - 'zero': Weights are set to zero. 
        randomize: Boolean.
            Whether to randomize the instance list during each epoch.
        """

        loss_func = "mse"
        optimizer_name = "bayesian"
        super(self.__class__, self).__init__(in_dim, hidden_dim_list, out_dim, 
            activation_func_list, optimizer_params, loss_func, optimizer_name)
   
        self.mu = optimizer_params["mu"]
        self.mult_factor = optimizer_params["mult_factor"]
        self.gradnorm_thres = optimizer_params["gradnorm_thres"]
        self.randomize = randomize

    


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
        alpha_div_beta_list: List of float.
            The value of alpha divided by beta at each epoch.
        gamma_list: List of float.
            The gamma (number of effective parameters) values at each epoch. 
        alpha_list: List of float.
            The value of alpha at each epoch.
        beta_list: List of float.
            The value of beta at each epoch. 
        ssq_term_list: List of float.
            The value of the sum of square term at each epoch.
        w_rg_term_list: List of float.
            The value of the L2 regularization term at each epoch.    
        """
   
        print("Begin training")
        n_inst = X_train.shape[0]
        n_total_params = get_total_number_of_parameters(self)
        identity_n_total_params = np.identity(n_total_params)
        
        loss_list = []
        alpha_div_beta_list = []
        gamma_list = []
        alpha_list = []
        beta_list = []
        ssq_term_list = []
        w_rg_term_list = []
        
        gamma = None # Parameters for the Bayesian regularization procedure
        alpha = None
        beta = None
        
        for epoch_idx in range(1, n_epoch+1):
            ssq_term = 0 # Sum of square term
            random_idx_list = [idx for idx in range(n_inst)]
            if self.randomize is True:
                random.shuffle(random_idx_list)
            vmat = np.zeros(self.out_dim*n_inst)
            vmat_idx = 0
            out_list_all = []
            for inst_idx in range(n_inst):
                x_inst = X_train[random_idx_list[inst_idx], :]
                y_inst = y_train[random_idx_list[inst_idx]]
                
                if np.size(y_inst) == 1:
                    if np.isscalar(y_inst):
                        y_inst = np.array([[y_inst]])
                    else:
                        y_inst = y_inst[:, np.newaxis]                
                    
                # Forward pass
                out, out_list = self.eval(x_inst)
                
                for j1_idx in range(self.out_dim):
                    vmat[vmat_idx] = y_inst[j1_idx]-out[j1_idx]
                    vmat_idx += 1

                out_list_all.append(out_list)
                                
                # Calculate the sum square error
                ssq_term += np.sum((y_inst-out)**2)

            # Calculate the weight regularization term (here it is assumed to be L2)
            w_rg_term = 0
            for layer_idx in range(self.n_layer):
                weight = self.layer_list[layer_idx]["weight"]
                bias = self.layer_list[layer_idx]["bias"]
                w_rg_term += np.sum(weight**2)
                w_rg_term += np.sum(bias**2)
            
            ssq_term_list.append(ssq_term)
            w_rg_term_list.append(w_rg_term)
            
            # Take one step of the Levenberg-Marquardt algorithm 
            if gamma is None:
                gamma = n_total_params
                alpha = gamma / (2.0*w_rg_term)
                beta = (n_inst*self.layer_list[-1]["out_curr_dim"]-gamma) / (2.0*ssq_term)
            
            logging.debug("Gamma: %f" % (gamma))
            logging.debug("Alpha: %f" % (alpha))
            logging.debug("Beta: %f" % (beta))
            
            alpha_div_beta_list.append(alpha / (1.0*beta))
            gamma_list.append(gamma)
            alpha_list.append(alpha)
            beta_list.append(beta)
            
            loss = beta*ssq_term + alpha*w_rg_term            
            print("Epoch %d, loss value: %f" % (epoch_idx, loss))
            
            t1 = time.time()
            # Calculate Marquadt sensitivity matrices
            m_ss_list_all = {
            }           
            
            # for layer_idx in range(0, self.n_layer):
                # m_ss_list_all[layer_idx] = None
                # m_ss_list_all[layer_idx] = []
            
            for layer_idx in reversed(range(0, self.n_layer)):
                if layer_idx == (self.n_layer-1):
                    # Last layer, initialized based on the outputs
                    m_ss_list = []
                    for inst_idx in range(n_inst):
                        out = out_list_all[inst_idx][-1]
                        fgrad = eval_activation_func_gradient(out, self.layer_list[-1]["activation_func"])
                        Fdot = np.diag(fgrad)
                        m_ss_inst = -Fdot
                        if (np.size(m_ss_inst.shape) == 1) and (np.isscalar(m_ss_inst) is False):
                            m_ss_inst = m_ss_inst[:, np.newaxis]
                        m_ss_list.append(m_ss_inst)
                    # set_trace()
                    m_ss_list_all[layer_idx] = m_ss_list
                else:
                    # For other layers, update from the next layer
                    m_ss_list = []
                    for inst_idx in range(n_inst):
                        out = out_list_all[inst_idx][layer_idx]
                        fgrad = eval_activation_func_gradient(out, self.layer_list[layer_idx]["activation_func"])
                        if (np.size(fgrad.shape) == 1) or (np.size(fgrad) == 1):
                            Fdot = np.diag(fgrad)
                        else:
                            Fdot = np.diag(fgrad.squeeze())
                        m_ss_inst = np.dot(Fdot, np.transpose(self.layer_list[layer_idx+1]["weight"]))
                        m_ss_inst = np.dot(m_ss_inst, m_ss_list_all[layer_idx+1][inst_idx])
                        if np.isscalar(m_ss_inst):
                            m_ss_inst = np.array([[m_ss_inst]])
                        if (np.size(m_ss_inst.shape) == 1) and (np.isscalar(m_ss_inst) is False):
                            m_ss_inst = m_ss_inst[:, np.newaxis]

                        m_ss_list.append(m_ss_inst)
                    m_ss_list_all[layer_idx] = m_ss_list

            m_ss_all = {
            }
            for layer_idx in range(0, self.n_layer):
                m_ss_all[layer_idx] = np.concatenate(m_ss_list_all[layer_idx], axis=1)
            t2 = time.time()
            print(f"Calculate Marquadt sensitivity matrices: {t2-t1} seconds")
            
            t1 = time.time()
            # Calculate the Jacobian matrix
            # jacobian = np.zeros((self.out_dim*n_inst, n_total_params))
            # for inst_idx in range(n_inst):
                # for out_dim_idx in range(self.out_dim):
                    # row_idx = inst_idx*self.out_dim + out_dim_idx
                    # col_idx = 0
                    # for layer_idx in range(0, self.n_layer):
                        # Weight
                        # for i1_idx in range(self.layer_list[layer_idx]["out_curr_dim"]):
                            # for j1_idx in range(self.layer_list[layer_idx]["in_curr_dim"]):
                                # if layer_idx == 0:
                                    # a_m_minus_one = X_train[random_idx_list[inst_idx], j1_idx]
                                # else:
                                    # a_m_minus_one = out_list_all[inst_idx][layer_idx-1][j1_idx]
                                # jacobian[row_idx, col_idx] = m_ss_all[layer_idx][i1_idx, row_idx]*a_m_minus_one
                                # col_idx += 1
                                
                        # Bias
                        # for i1_idx in range(self.layer_list[layer_idx]["out_curr_dim"]):
                            # jacobian[row_idx, col_idx] = m_ss_all[layer_idx][i1_idx, row_idx]
                            # col_idx += 1
            # jacobian = self.get_jacobian_matrix(X_train, n_inst, n_total_params, random_idx_list,
                # out_list_all, m_ss_all)
            jacobian = get_jacobian_matrix(X_train, n_inst, n_total_params, self.n_layer, self.layer_list,
                random_idx_list, out_list_all, m_ss_all, self.out_dim)
            t2 = time.time()
            print(f"Calculate the Jacobian matrix: {t2-t1} seconds")
            transp_jac_mult_jac = np.dot(np.transpose(jacobian), jacobian) 
            grad_div_2 = np.dot(np.transpose(jacobian), vmat)  
            while True:
                t1 = time.time()
                # Update the weights based on the Jacobian matrix
                t3 = time.time()
                # mat = np.dot(np.transpose(jacobian), jacobian)+self.mu*np.identity(n_total_params)
                # mat = np.dot(np.transpose(jacobian), jacobian)+self.mu*identity_n_total_params
                mat = transp_jac_mult_jac+self.mu*identity_n_total_params                
                inv_mat = np.linalg.inv(mat)
                # grad_div_2 = np.dot(np.transpose(jacobian), vmat)                
                delta_w = -np.dot(inv_mat, grad_div_2)                
                w_curr = flatten_into_vector(self)           
                w_new = w_curr + delta_w            
                t4 = time.time()
                print(f"Update the weights based on the Jacobian matrix: {t4-t3} seconds")
                
                # Update the mu parameter
                t3 = time.time()
                unflatten_from_vector(self, w_new, is_grad=False)
                ssq_term_new = 0
                for inst_idx in range(n_inst):
                    x_inst = X_train[random_idx_list[inst_idx], :]
                    y_inst = y_train[random_idx_list[inst_idx]]
                    
                    if np.size(y_inst) == 1:
                        if np.isscalar(y_inst):
                            y_inst = np.array([[y_inst]])
                        else:
                            y_inst = y_inst[:, np.newaxis]                
                        
                    # Forward pass
                    out_new, _ = self.eval(x_inst)                
                    ssq_term_new += np.sum((y_inst-out_new)**2)
                
                w_rg_term_new = 0
                for layer_idx in range(self.n_layer):
                    weight = self.layer_list[layer_idx]["weight"]
                    bias = self.layer_list[layer_idx]["bias"]
                    w_rg_term_new += np.sum(weight**2)
                    w_rg_term_new += np.sum(bias**2)
                t4 = time.time()
                print(f"mu parameter: {t4-t3} seconds")
                
                loss_new = beta*ssq_term_new + alpha*w_rg_term_new
                t2 = time.time()
                print(f"One step of while loop takes: {t2-t1} seconds")
                if loss_new < loss:
                    # In this case we reduce mu and go to the next epoch                   
                    self.mu /= self.mult_factor
                    logging.debug("Reduce mu to new value: %f" % (self.mu))
                    if self.mu < 1e-20:
                        self.mu = 1e-20
                    break
                else:
                    # In this case we increase mu, return to the current position and try again
                    self.mu *= self.mult_factor
                    unflatten_from_vector(self, w_curr)
                    logging.debug("Increase mu to new value: %f, return to the current position and try again" % (self.mu))
                    if self.mu >= 1e10:
                        break
                    continue
            
            # After the step, we now update the Bayesian regularization parameters
            # hessian = 2*beta*np.dot(np.transpose(jacobian), jacobian) + 2*alpha*np.identity(n_total_params)
            hessian = 2*beta*transp_jac_mult_jac + 2*alpha*identity_n_total_params
            gamma = n_total_params - 2*alpha*np.trace(np.linalg.inv(hessian))
            alpha = gamma / (2.0*w_rg_term)
            beta = (n_inst*self.layer_list[-1]["out_curr_dim"]-gamma) / (2.0*ssq_term) 
            
            loss_list.append(loss)
            
            gradnorm = np.sqrt(np.sum(grad_div_2**2))
            logging.debug("Gradnorm: %f" % (gradnorm))
            if gradnorm < self.gradnorm_thres:            
                logging.debug("Gradnorm smaller than threshold %f" % (self.gradnorm_thres))
                print("Finish training")
                break
        
        return loss_list, alpha_div_beta_list, gamma_list, alpha_list, beta_list, ssq_term_list, w_rg_term_list

def create_data():    
    T = 2
    X_train = np.linspace(-1, 1, 201)
    y_train = np.sin(2 * np.pi * X_train / T)    
    X_train = X_train[:, np.newaxis]    
    y_train = y_train[:, np.newaxis]   
    return (X_train, y_train)

@log_time
@show_info       
def test1():
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 10,
        "gradnorm_thres": 0.1,
    }
    net = BayesianRegularizedNeuralNetwork(
        1, [20], 1, ["sigmoid", "linear"], optimizer_params,
    )
    # n_epoch = 20
    n_epoch = 6
    # set_trace()
    loss_list, alpha_div_beta_list, gamma_list, alpha_list, beta_list, ssq_term_list, w_rg_term_list = net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out, _ = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network using Bayesian regularization algorithm.")
    plt.show()
    
    plt.plot([idx for idx in range(len(alpha_div_beta_list))], alpha_div_beta_list, color="red")
    plt.title("Alpha divided by beta.")
    plt.show()
    
    plt.plot([idx for idx in range(len(gamma_list))], gamma_list, color="red")
    plt.title("Gamma.")
    plt.show()
    
    plt.plot([idx for idx in range(len(alpha_list))], alpha_list, color="red")
    plt.title("Alpha.")
    plt.show()

    plt.plot([idx for idx in range(len(beta_list))], beta_list, color="red")
    plt.title("Beta.")
    plt.show()

    plt.plot([idx for idx in range(len(ssq_term_list))], ssq_term_list, color="red")
    plt.title("Sum of square term.")
    plt.show()
   
    plt.plot([idx for idx in range(len(w_rg_term_list))], w_rg_term_list, color="red")
    plt.title("Weight regularization term.")
    plt.show()
    
    print("Final alpha/beta value: %f" % (alpha_div_beta_list[-1]))
    print("Final gamma value: %f" % (gamma_list[-1]))

@log_time
@show_info 
def test2():
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 10,
        "gradnorm_thres": 0.1,
    }
    net = BayesianRegularizedNeuralNetwork(
        1, [50, 50], 1, ["LeakyRelu", "LeakyRelu", "linear"], optimizer_params,
    )
    n_epoch = 100
    loss_list, alpha_div_beta_list, gamma_list, alpha_list, beta_list, ssq_term_list, w_rg_term_list = net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out, _ = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network using Bayesian regularization algorithm.")
    plt.show()
    
    plt.plot([idx for idx in range(len(alpha_div_beta_list))], alpha_div_beta_list, color="red")
    plt.title("Alpha divided by beta.")
    plt.show()
    
    plt.plot([idx for idx in range(len(gamma_list))], gamma_list, color="red")
    plt.title("Gamma.")
    plt.show()
    
    plt.plot([idx for idx in range(len(alpha_list))], alpha_list, color="red")
    plt.title("Alpha.")
    plt.show()

    plt.plot([idx for idx in range(len(beta_list))], beta_list, color="red")
    plt.title("Beta.")
    plt.show()

    plt.plot([idx for idx in range(len(ssq_term_list))], ssq_term_list, color="red")
    plt.title("Sum of square term.")
    plt.show()
   
    plt.plot([idx for idx in range(len(w_rg_term_list))], w_rg_term_list, color="red")
    plt.title("Weight regularization term.")
    plt.show()
    
    print("Final alpha/beta value: %f" % (alpha_div_beta_list[-1]))
    print("Final gamma value: %f" % (gamma_list[-1]))

@log_time
@show_info 
def test3():    
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 25, # Set to 10: Oscillates between high mu values from epoch 50 onwards
        "gradnorm_thres": 0.1,
    }
    net = BayesianRegularizedNeuralNetwork(
        1, [50, 50, 50], 1, ["LeakyRelu", "LeakyRelu", "LeakyRelu", "linear"], optimizer_params,
    )
    n_epoch = 100
    loss_list, alpha_div_beta_list, gamma_list, alpha_list, beta_list, ssq_term_list, w_rg_term_list = net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out, _ = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network using Bayesian regularization algorithm.")
    plt.show()
    
    plt.plot([idx for idx in range(len(alpha_div_beta_list))], alpha_div_beta_list, color="red")
    plt.title("Alpha divided by beta.")
    plt.show()
    
    plt.plot([idx for idx in range(len(gamma_list))], gamma_list, color="red")
    plt.title("Gamma.")
    plt.show()
    
    plt.plot([idx for idx in range(len(alpha_list))], alpha_list, color="red")
    plt.title("Alpha.")
    plt.show()

    plt.plot([idx for idx in range(len(beta_list))], beta_list, color="red")
    plt.title("Beta.")
    plt.show()

    plt.plot([idx for idx in range(len(ssq_term_list))], ssq_term_list, color="red")
    plt.title("Sum of square term.")
    plt.show()
   
    plt.plot([idx for idx in range(len(w_rg_term_list))], w_rg_term_list, color="red")
    plt.title("Weight regularization term.")
    plt.show()
    
    print("Final alpha/beta value: %f" % (alpha_div_beta_list[-1]))
    print("Final gamma value: %f" % (gamma_list[-1]))
    

def main():
    # test1()    
    # test2()    
    test3()


if __name__ == "__main__":
    main()
