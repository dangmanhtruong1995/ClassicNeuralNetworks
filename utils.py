import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
import copy
from time import time
import functools

import warnings
warnings.filterwarnings('ignore')

def init_matrix(n_in, n_out, method="kaiming"):
    # set_trace()
    if method == "kaiming":
        limit = np.sqrt(6.0 / float(n_in + n_out))
        mat = np.random.uniform(low=-limit, high=limit, size=(n_in, n_out)).transpose()
    elif method == "normal":
        mat = np.random.normal(size=(n_out, n_in))
    return mat
    

def get_total_number_of_parameters(net):
    n_dim = 0
    for layer in net.layer_list:
        n_dim += (layer["in_curr_dim"]+1)*layer["out_curr_dim"] # Each layer has weights and bias
    return n_dim
 
def flatten_into_vector(net, is_grad=False):
    if is_grad is False:
        w_layer_name = "weight"
        b_layer_name = "bias"
    else:
        w_layer_name = "weight_grad"
        b_layer_name = "bias_grad"
    
    n_dim = get_total_number_of_parameters(net)    
    arr = np.zeros(n_dim)
    
    last_idx = 0
    first_idx = last_idx 
    for layer in net.layer_list:
        in_curr_dim = layer["in_curr_dim"]
        out_curr_dim = layer["out_curr_dim"]
        
        last_idx = first_idx + out_curr_dim*in_curr_dim
        arr[first_idx:last_idx] = layer[w_layer_name].flatten()
        
        first_idx = last_idx
        last_idx = first_idx + out_curr_dim

        arr[first_idx:last_idx] = layer[b_layer_name].flatten()
        first_idx = last_idx
    
    return arr

def unflatten_from_vector(net, arr, is_grad=False):
    if is_grad is False:
        w_layer_name = "weight"
        b_layer_name = "bias"
    else:
        w_layer_name = "weight_grad"
        b_layer_name = "bias_grad"
    
    n_dim = get_total_number_of_parameters(net) 
    
    last_idx = 0
    first_idx = last_idx 
    for layer in net.layer_list:
        in_curr_dim = layer["in_curr_dim"]
        out_curr_dim = layer["out_curr_dim"]
        
        last_idx = first_idx + out_curr_dim*in_curr_dim
        layer[w_layer_name] = arr[first_idx:last_idx].reshape((out_curr_dim, in_curr_dim))
        
        first_idx = last_idx
        last_idx = first_idx + out_curr_dim
        
        layer[b_layer_name] = arr[first_idx:last_idx].reshape((out_curr_dim))
        first_idx = last_idx
    
def eval_activation_func(x, activation_func, leaky_alpha=0.01):
    if activation_func == "sigmoid":
        return 1/(1+np.exp(-x))
    elif activation_func == "linear":
        return x
    elif activation_func == "tanh":
        return np.tanh(x)
    elif activation_func == "hardlim":
        return (x >= 0).astype(np.float32)
    elif activation_func == "square":
        return np.square(x)
    elif activation_func == "relu":
        # This function is also known with the name Positive Linear tranfer function (poslin)
        out = copy.deepcopy(x)
        if np.isscalar(out):
            if out < 0:
                out = 0
        else:
            out[out < 0] = 0
        return out
    elif activation_func == "LeakyRelu":
        out = copy.deepcopy(x)
        if np.isscalar(out):
            if out < 0:
                out *= leaky_alpha
        else:
            out[out < 0] *= leaky_alpha
        return out
    elif activation_func == "elu":
        out = copy.deepcopy(x)
        if np.isscalar(out):
            if out < 0:
                out = leaky_alpha*(np.exp(out)-1)
        else:
            out[out < 0] *= leaky_alpha*(np.exp(out[out < 0])-1)
        return out
    else:
        raise Exception("In eval_activation_func: Unimplemented function '%s'" % (activation_func))

def eval_activation_func_gradient(out, activation_func, leaky_alpha=0.01):
    if activation_func == "sigmoid":
        return out*(1-out)        
    elif activation_func == "linear":
        return np.ones_like(out)
    elif activation_func == "square":
        return 2*np.sqrt(out) # TODO: Rewrite to make more efficient
    elif activation_func == "relu":
        grad = copy.deepcopy(out)
        if np.isscalar(grad):
            if grad > 0:
                grad = 1
            else:
                grad = 0
        else:        
            grad[out > 0] = 1
            grad[out <= 0] = 0
        return grad
    elif activation_func == "LeakyRelu":
        grad = copy.deepcopy(out)
        if np.isscalar(grad):
            if grad > 0:
                grad = 1
            else:
                grad = leaky_alpha
        else:        
            grad[out > 0] = 1
            grad[out <= 0] = leaky_alpha
        return grad
    elif activation_func == "elu":
        grad = copy.deepcopy(out)
        if np.isscalar(grad):
            if grad > 0:
                grad = 1
            else:
                grad = leaky_alpha*np.exp(out)
        else:        
            grad[out > 0] = 1
            grad[out <= 0] = leaky_alpha*np.exp(out[out <= 0])
        return grad
    # elif activation_func == "tanh":
        # return np.tanh(x)
    # elif activation_func == "hardlim":
        # return (x >= 0).astype(np.float32)
    # else:
        raise Exception("In eval_activation_func: Unimplemented function '%s'" % (activation_func))

def eval_loss_func(y_true, y_pred, loss_func="mse"):
    if loss_func == "mse":
        loss = 0
        # if np.size(y_true) == 1:
        if (np.size(y_true) == 1) and (np.size(y_true.shape)==1):
            y_true = y_true[:, np.newaxis]
        # set_trace()
        n_inst = y_true.shape[0]
        for inst_idx in range(n_inst):
            for dim_idx in range(y_pred.shape[1]):
                loss += (y_true[inst_idx, dim_idx] - y_pred[inst_idx, dim_idx])**2
        try:
            loss /= (1.0*n_inst)
        except:
            set_trace()
        return loss
    elif loss_func == "crossentropy":
        pass
    else:
        raise Exception("In eval_loss_func: Unimplemented function '%s'" % (loss_func))        
        
def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time()
        original_result = func(*args, **kwargs)
        t2 = time()
        print("")
        print("Running function named %s took %f seconds." % (func.__name__, t2-t1))
        print("")
        return original_result
    return wrapper
   
def show_info(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("")
        print("")
        print("")
        print("--------------------------------------")
        print("Begin function %s" % (func.__name__))
        print("--------------------------------------")
        result = func(*args, **kwargs)
        return result
    return wrapper
        
