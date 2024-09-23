import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
import random
import time
from copy import deepcopy
from pprint import pprint
import itertools

from config import get_config

import warnings
warnings.filterwarnings('ignore')

# import tensorflow as tf

import logging

logging.basicConfig(level=get_config()["debug_mode"])
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class AdamOptimizerRNN:
    """ Adam optimizer, described in [1] (I have to write a separate one for RNN due to messy code).
    
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
    step(epoch_idx, model)
        Run one epoch.     
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
        pass

    def step(self, epoch_idx, model):
        if epoch_idx == 0:
            # Initialize 
            lr_curr = self.learning_rate
            first_moment_iw_grad_dict = {}
            first_moment_lw_grad_dict = {}
            first_moment_bias_grad_dict = {}
   
            second_moment_iw_grad_dict = {}
            second_moment_lw_grad_dict = {}
            second_moment_bias_grad_dict = {}

            for (layer_dest_name, layer_name) in model.iw_grad:
                first_moment_iw_grad_dict[(layer_dest_name, layer_name)] = np.zeros_like(model.iw_grad[(layer_dest_name, layer_name)])
                second_moment_iw_grad_dict[(layer_dest_name, layer_name)] = np.zeros_like(model.iw_grad[(layer_dest_name, layer_name)])
            
            for (layer_dest_name, layer_name) in model.lw_grad:
                first_moment_lw_grad_dict[(layer_dest_name, layer_name)] = np.zeros_like(model.lw_grad[(layer_dest_name, layer_name)])
                second_moment_lw_grad_dict[(layer_dest_name, layer_name)] = np.zeros_like(model.lw_grad[(layer_dest_name, layer_name)])
            
            for layer_name in model.bias_grad:
                first_moment_bias_grad_dict[layer_name] = np.zeros_like(model.bias_grad[layer_name])
                second_moment_bias_grad_dict[layer_name] = np.zeros_like(model.bias_grad[layer_name])

        else:
            first_moment_iw_grad_dict = self.first_moment_iw_grad_dict
            first_moment_lw_grad_dict = self.first_moment_lw_grad_dict
            first_moment_bias_grad_dict = self.first_moment_bias_grad_dict

            second_moment_iw_grad_dict = self.second_moment_iw_grad_dict
            second_moment_lw_grad_dict = self.second_moment_lw_grad_dict
            second_moment_bias_grad_dict = self.second_moment_bias_grad_dict
        
        # Update first and second moment estimates
        for (layer_dest_name, layer_name) in model.iw_grad:
            first_moment_iw_grad_dict[(layer_dest_name, layer_name)] = self.beta1*first_moment_iw_grad_dict[(layer_dest_name, layer_name)]+(1-self.beta1)*model.iw_grad[(layer_dest_name, layer_name)]
            second_moment_iw_grad_dict[(layer_dest_name, layer_name)] = self.beta2*second_moment_iw_grad_dict[(layer_dest_name, layer_name)]+(1-self.beta2)*np.square(model.iw_grad[(layer_dest_name, layer_name)])

        for (layer_dest_name, layer_name) in model.lw_grad:
            first_moment_lw_grad_dict[(layer_dest_name, layer_name)] = self.beta1*first_moment_lw_grad_dict[(layer_dest_name, layer_name)]+(1-self.beta1)*model.lw_grad[(layer_dest_name, layer_name)]
            second_moment_lw_grad_dict[(layer_dest_name, layer_name)] = self.beta2*second_moment_lw_grad_dict[(layer_dest_name, layer_name)]+(1-self.beta2)*np.square(model.lw_grad[(layer_dest_name, layer_name)])

        for layer_name in model.bias_grad:
            first_moment_bias_grad_dict[layer_name] = self.beta1*first_moment_bias_grad_dict[layer_name]+(1-self.beta1)*model.bias_grad[layer_name]
            second_moment_bias_grad_dict[layer_name] = self.beta2*second_moment_bias_grad_dict[layer_name]+(1-self.beta2)*np.square(model.bias_grad[layer_name])

        # Bias correction
        if epoch_idx > 0:
            lr_curr = self.learning_rate*(np.sqrt(1-np.power(self.beta2, epoch_idx))/(1-np.power(self.beta1, epoch_idx)))

        # Update the network parameters
        for (layer_dest_name, layer_name) in model.iw_grad:
            model.iw[(layer_dest_name, layer_name)] -= (lr_curr*first_moment_iw_grad_dict[(layer_dest_name, layer_name)] / (np.sqrt(second_moment_iw_grad_dict[(layer_dest_name, layer_name)])+self.epsilon)) 
        
        for (layer_dest_name, layer_name) in model.lw_grad:
            model.lw[(layer_dest_name, layer_name)] -= (lr_curr*first_moment_lw_grad_dict[(layer_dest_name, layer_name)] / (np.sqrt(second_moment_lw_grad_dict[(layer_dest_name, layer_name)])+self.epsilon)) 

        for layer_name in model.bias_grad:
            model.bias[layer_name] -= (lr_curr*first_moment_bias_grad_dict[layer_name] / (np.sqrt(second_moment_bias_grad_dict[layer_name])+self.epsilon)) 

        self.first_moment_iw_grad_dict = first_moment_iw_grad_dict
        self.first_moment_lw_grad_dict = first_moment_lw_grad_dict
        self.first_moment_bias_grad_dict = first_moment_bias_grad_dict

        self.second_moment_iw_grad_dict = second_moment_iw_grad_dict
        self.second_moment_lw_grad_dict = second_moment_lw_grad_dict
        self.second_moment_bias_grad_dict = second_moment_bias_grad_dict
        

def check_weights_and_gradient_shapes(func):        
    def _wrapper(self, *args, **kwargs):
        out = func(self, *args,**kwargs)
        self._check_shapes()
        return out
    return _wrapper

def sigmoid(x1):
    return 1.0/(1.0+np.exp(-x1))

def eval_loss_func_rnn(gt_dict, pred_dict, loss_func="SSE"):
    if loss_func == "SSE":
        # Sum of square error
        loss_dict = {}
        loss_grad_dict = {}
        for key in gt_dict:
            try:
                loss_dict[key] = np.sum((gt_dict[key]-pred_dict[key])**2)
            except:
                set_trace() 
            loss_grad_dict[key] = (-2)*(gt_dict[key]-pred_dict[key])
    elif loss_func == "MSE":
        loss_dict = {}
        loss_grad_dict = {}
        for key in gt_dict:
            try:
                loss_dict[key] = np.sum((gt_dict[key]-pred_dict[key])**2)/(1.0*gt_dict[key].shape[0])
            except:
                set_trace() 
            loss_grad_dict[key] = (-2)*(gt_dict[key]-pred_dict[key])
            loss_grad_dict[key] /= (1.0*gt_dict[key].shape[0])

    elif loss_func == "binary_cross_entropy":
        # For classification, the loss is only calculated at the last time step, and the 
        # loss for the previous time steps are set to 0.
        loss_dict = {}
        loss_grad_dict = {}
        for key in gt_dict:
            try:
                print("try")
                pred = pred_dict[key][:, -1, :]
                one_hot_targets = np.eye(np.max(gt_dict[key])+1)[gt_dict[key]]
                one_hot_targets = one_hot_targets[:, 0, :, :]
                pred_sigmoid = sigmoid(pred)
                pred_two_cls = np.zeros((pred_sigmoid.shape[0], pred_sigmoid.shape[1], 2))
                pred_two_cls[:, :, 0] = pred_sigmoid
                pred_two_cls[:, :, 1] = 1 - pred_sigmoid

                pred_two_cls = pred_two_cls[:, 0, :]
                one_hot_targets = one_hot_targets[:, 0, :]
                n_inst = pred_two_cls.shape[0]
                n_cls = one_hot_targets.shape[1]
                cross_entropy_val = 0
#                for inst_idx in range(n_inst):
#                    cross_entropy_val -= np.sum(one_hot_targets[inst_idx, :]*np.log(pred_two_cls[inst_idx, :]))
                cross_entropy_val = np.sum(one_hot_targets*pred_two_cls)
                cross_entropy_val = -cross_entropy_val
                cross_entropy_val /= (1.0*n_inst)
                loss_dict[key] = cross_entropy_val

                loss_grad = np.zeros_like(pred)
                for inst_idx in range(n_inst):
                    loss_grad[inst_idx] = (pred_two_cls[inst_idx, 0]-gt_dict[key][inst_idx, 0, 0])/(pred_two_cls[inst_idx, 0]*(1-pred_two_cls[inst_idx, 0]))
#                loss_grad = (pred_two_cls-gt_dict[key][:, 0, 0]) / (pred_two_cls*(1-pred_two_cls))
                loss_grad_dict[key] = np.zeros_like(pred_dict[key])
                loss_grad_dict[key][:, -1, :] = loss_grad
#                set_trace() 
            except:
                set_trace() 
        pass
    else:
        raise Exception("In eval_loss_func_rnn: Unimplemented loss function '%s'" % (loss_func))   
    return loss_dict, loss_grad_dict
