import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
import random

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

class LevenbergMarquadtNeuralNetwork(GenericFeedforwardNeuralNetwork):
    """ Neural network using the Levenberg-Marquadt algorithm for training.
    
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
    loss_prev: None or float.
        The loss value of the previous epoch.
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
        Train the network
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
        init_type: String.
            Initialization type. Default is 'random'. 
            - 'random': Weights are randomly initialized.
            - 'zero': Weights are set to zero. 
        randomize: Boolean.
            Whether to randomize the instance list during each epoch.
        """

        loss_func = "mse"
        optimizer_name = "Levenberg-Marquadt"
        super(self.__class__, self).__init__(in_dim, hidden_dim_list, out_dim, 
            activation_func_list, optimizer_params, loss_func, optimizer_name)

        self.mu = optimizer_params["mu"]
        self.mult_factor = optimizer_params["mult_factor"]
        self.loss_prev = None
        self.randomize = randomize
        
        self.gradnorm_thres = 0.1


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
        n_total_params = get_total_number_of_parameters(self)
        identity_n_total_params = np.identity(n_total_params)
        self.is_first_epoch = True
        loss_list = []
        # set_trace()
        for epoch_idx in range(1, n_epoch+1):
            loss = 0
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
                                
                # Calculate the loss
                loss += self.loss(y_inst, out)
            
            # Calculate Marquadt sensitivity matrices
            m_ss_list_all = {
            }
            for layer_idx in range(0, self.n_layer):
                m_ss_list_all[layer_idx] = None
            
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
                
            # Calculate the Jacobian matrix
            jacobian = np.zeros((self.out_dim*n_inst, n_total_params))
            for inst_idx in range(n_inst):
                for out_dim_idx in range(self.out_dim):
                    row_idx = inst_idx*self.out_dim + out_dim_idx
                    col_idx = 0
                    for layer_idx in range(0, self.n_layer):
                        # Weight
                        for i1_idx in range(self.layer_list[layer_idx]["out_curr_dim"]):
                            for j1_idx in range(self.layer_list[layer_idx]["in_curr_dim"]):
                                if layer_idx == 0:
                                    a_m_minus_one = X_train[random_idx_list[inst_idx], j1_idx]
                                else:
                                    a_m_minus_one = out_list_all[inst_idx][layer_idx-1][j1_idx]
                                jacobian[row_idx, col_idx] = m_ss_all[layer_idx][i1_idx, row_idx]*a_m_minus_one
                                col_idx += 1
                                
                        # Bias
                        for i1_idx in range(self.layer_list[layer_idx]["out_curr_dim"]):
                            jacobian[row_idx, col_idx] = m_ss_all[layer_idx][i1_idx, row_idx]
                            col_idx += 1
            
            transp_jac_mult_jac = np.dot(np.transpose(jacobian), jacobian) 
            grad_div_2 = np.dot(np.transpose(jacobian), vmat) 
            while True:
                # Update the weights based on the Jacobian matrix
                # mat = np.dot(np.transpose(jacobian), jacobian)+self.mu*np.identity(n_total_params)
                mat = transp_jac_mult_jac+self.mu*identity_n_total_params 
                inv_mat = np.linalg.inv(mat)
                # delta_w = np.dot(inv_mat, np.transpose(jacobian))
                # delta_w = -np.dot(delta_w, vmat)
                # grad_div_2 = np.dot(np.transpose(jacobian), vmat)
                delta_w = -np.dot(inv_mat, grad_div_2)                
                w_curr = flatten_into_vector(self)
                w_new = w_curr + delta_w            
                
                # Update the mu parameter
                unflatten_from_vector(self, w_new, is_grad=False)
                loss_new = 0
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
                    loss_new += self.loss(y_inst, out_new)
                
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
            
            gradnorm = np.sqrt(np.sum(grad_div_2**2))
            # set_trace()
            logging.debug("Gradnorm: %f" % (gradnorm))
            if gradnorm < self.gradnorm_thres:            
                logging.debug("Gradnorm smaller than threshold %f" % (self.gradnorm_thres))
                print("Finish training")
                break
            
            # loss /= (1.0*n_inst)
            print("Epoch %d, loss value: %f" % (epoch_idx, loss))
            
        print("Finish training")
        
        return loss_list

def create_data(): 
    X_train = np.arange(-2, 3, 0.01)
    X_train = X_train[:, np.newaxis]
    y_train = np.zeros(X_train.shape[0])
    for inst_idx in range(X_train.shape[0]):
        inst = X_train[inst_idx, :]
        y_train[inst_idx] = 1+np.sin(1.5*np.pi*inst)
    y_train = y_train[:, np.newaxis]    
    # set_trace()
    return (X_train, y_train)

@log_time
@show_info
def test1():
    # For testing 
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 10,
    }
    n_epoch = 1
    net = LevenbergMarquadtNeuralNetwork(
        1, [1], 1, ["square", "linear"], optimizer_params, init_type="zero", randomize=False,
    )
    net.layer_list[0]["weight"] = np.array(
        [[1],
        ])
    net.layer_list[0]["bias"] = np.array(
        [[0],
        ])
    net.layer_list[1]["weight"] = np.array(
        [[2],
        ])
    net.layer_list[1]["bias"] = np.array(
        [[1],
        ])
    X_train = np.array([
        [1],
        [2],
    ])
    y_train = np.array([
        1,
        2,
    ])
    net.train(X_train, y_train, n_epoch)

@log_time
@show_info
def test2():
    # More than 50 epochs for convergence, as compared to 100000 epochs by gradient descent,
    # and around 3000 epochs by conjugate gradient algorithm. 
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 5
    }
    net = LevenbergMarquadtNeuralNetwork(
        1, [10], 1, ["sigmoid", "linear"], optimizer_params,
    )
    n_epoch = 100
    # n_epoch = 50
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network (1-10-1, sigmoid-linear) using Levenberg-Marquadt algorithm.")
    plt.show()

@log_time
@show_info
def test3():
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 5
    }
    net = LevenbergMarquadtNeuralNetwork(
        1, [50, 50], 1, ["sigmoid", "sigmoid", "linear"], optimizer_params,
    )
    n_epoch = 100
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network (1-50-50-1, sigmoid-sigmoid-linear) using Levenberg-Marquadt algorithm.")
    plt.show()

@log_time
@show_info
def test4():
    # L-M algorithm but this time the network consists of ReLU activations
    # Requires around 10 epochs (sometimes converges poorly)
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 5
    }
    net = LevenbergMarquadtNeuralNetwork(
        1, [50, 50], 1, ["relu", "relu", "linear"], optimizer_params,
    )
    n_epoch = 100
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network (1-50-50-1, ReLU-ReLU-linear) using Levenberg-Marquadt algorithm.")
    plt.show()

@log_time
@show_info
def test5():
    # L-M algorithm but this time the network consists of Leaky ReLU activations
    # Only requires 10 epochs
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 5,        
    }
    net = LevenbergMarquadtNeuralNetwork(
        1, [50, 50], 1, ["LeakyRelu", "LeakyRelu", "linear"], optimizer_params,
    )
    n_epoch = 100
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network (1-50-50-1, LeakyReLU-LeakyReLU-linear) using Levenberg-Marquadt algorithm.")
    plt.show()

@log_time
@show_info
def test6():
    # L-M algorithm but this time the network consists of ELU activations
    # Overall training is more unstable compared to Leaky ReLU
    # 150 epochs and the result is still not as good
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 5,        
    }
    net = LevenbergMarquadtNeuralNetwork(
        1, [50, 50], 1, ["elu", "elu", "linear"], optimizer_params,
    )
    n_epoch = 100
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network (1-50-50-1, ELU-ELU-linear) using Levenberg-Marquadt algorithm.")
    plt.show()

@log_time
@show_info
def test7():
    # L-M algorithm but this time the network consists of a mixture of ReLU and 
    # Leaky ReLU activations
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 5,        
    }
    net = LevenbergMarquadtNeuralNetwork(
        1, [50, 50], 1, ["relu", "LeakyRelu", "linear"], optimizer_params,
    )
    n_epoch = 100
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network (1-50-50-1, ReLU-LeakyReLU-linear) using Levenberg-Marquadt algorithm.")
    plt.show()

@log_time
@show_info
def test8():
    # L-M algorithm but this time the network consists of Leaky ReLU activations
    # Network is 1-50-50-50-1
    # Only requires 10 epochs
    (X_train, y_train) = create_data()
    optimizer_params = {
        "mu": 0.01,
        "mult_factor": 5,        
    }
    net = LevenbergMarquadtNeuralNetwork(
        1, [50, 50, 50], 1, ["LeakyRelu", "LeakyRelu", "LeakyRelu", "linear"], optimizer_params,
    )
    n_epoch = 100
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red", label="Ground truth")
    plt.plot(base_list, y_pred, color="blue", label="Prediction")
    plt.legend(loc='upper left')
    plt.title("Neural network (1-50-50-50-1, LeakyReLU-LeakyReLU-LeakyReLU-linear) using Levenberg-Marquadt algorithm.")
    plt.show()
    
def main():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
    
if __name__ == "__main__":
    main()
