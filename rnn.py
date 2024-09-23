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

from utils import eval_activation_func, eval_activation_func_gradient, eval_loss_func, \
    flatten_into_vector, unflatten_from_vector, get_total_number_of_parameters, init_matrix
from rnn_utils import check_weights_and_gradient_shapes, eval_loss_func_rnn, AdamOptimizerRNN   

from config import get_config

import warnings
warnings.filterwarnings('ignore')

# import tensorflow as tf

import logging
logging.basicConfig(level=get_config()["debug_mode"])
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class InLayer:
    """ Class which stores basic information for the layer which would be used as input. 
        
    Methods
    -----------
    get_name()
        Return the layer name.    
    """
    
    def __init__(self, n_dim, builder, name="", act_func="sigmoid"):
        """
        Parameters
        ----------
        n_dim : Integer. 
            The layer dimension.
        builder: RNNBuilder object.
            The RNNBuilder object which will handle the building of the RNN based on the 
            layers and the connections between them that we provide.
        name : String (default is ""). 
            The layer name. 
        act_func: String (default is ""). 
            The name of the activation function.                       
        """
        
        if isinstance(n_dim, int) is False:
            raise Exception("In class %s: n_dim should be an int" % (self.__class__.__name__))
        if isinstance(builder, RNNBuilder) is False:
            raise Exception("In class %s: builder should be an instance of the RNNBuilder class" % (self.__class__.__name__))
        if isinstance(name, str) is False:
            raise Exception("In class %s: name should be a string" % (self.__class__.__name__))
        if isinstance(act_func, str)is False:
            raise Exception("In class %s: act_func should be a string" % (self.__class__.__name__))
                
        self.n_dim = n_dim
        self.act_func = act_func
        self.name = name
        self.builder = builder
        self.next = []
        
        if self.name in self.builder.layer_dict:
            raise Exception("In class %s: The name: %s has already been added to the computation graph." % (self.__class__.__name__, self.name))
        
        self.builder.layer_dict[self.name] = self
        self.builder.simulation_order.append(self.name)        
        self.builder.proper_input_layer_list.append(self.name)

    
    def get_name(self):
        """ Return the layer name.

        Parameters
        ---------------
        None.
        
        Returns
        ----------
        name: String.    
            The layer name
        """
        
        return self.name
 
 
class RNNLayer:
    """ Class which stores basic information for a RNN layer. 
        
    Methods
    -----------
    get_name()
        Return the layer name.
    __call__(layer_list)
        Add the connections between the current layers and the previous layers. 
        Each element in layer_list is of the form (layer, delay_list) where layer is 
        a layer which connects forward to RNNLayer, and delay_list is a list of 
        integers which denotes the list of delays between that layer and the current
        layer. 
    """
    
    def __init__(self, n_dim, builder,  name="", act_func="sigmoid", is_output=False):  
        """
        Parameters
        ----------
        n_dim : Integer. 
            The layer dimension.
        builder: RNNBuilder object.
            The RNNBuilder object which will handle the building of the RNN based on the 
            layers and the connections between them that we provide.
        name : String (default is ""). 
            The layer name. 
        act_func: String (default is ""). 
            The name of the activation function. 
        is_output: Boolean(Default: False). 
            If this argument is set to True, then the layer will be added to the output list, 
            and after the forward pass, the output of this layer will be returned in the output.               
        """
        
        if isinstance(n_dim, int) is False:
            raise Exception("In class %s: n_dim should be an int" % (self.__class__.__name__))    
        if isinstance(builder, RNNBuilder) is False:
            raise Exception("In class %s: builder should be an instance of the RNNBuilder class" % (self.__class__.__name__))
        if isinstance(name, str) is False:
            raise Exception("In class %s: name should be a string" % (self.__class__.__name__))
        if isinstance(act_func, str) is False:
            raise Exception("In class %s: act_func should be a string" % (self.__class__.__name__))
        
        self.n_dim = n_dim
        self.builder = builder
        self.name = name        
        self.act_func = act_func        
        self.is_output = is_output
        self.next = []
      
        if self.name in self.builder.layer_dict:
            raise Exception("In class %s: The name: %s has already been added to the computation graph." % (self.__class__.__name__, self.name))
                
        self.builder.layer_dict[self.name] = self
        self.builder.simulation_order.append(self.name)
        

    def get_name(self):
        """ Return the layer name.

        Parameters
        ---------------
        None.
        
        Returns
        ----------
        name: String.    
            The layer name
        """
        
        return self.name    


    def __call__(self, layer_in_list):
        """ Establish connections between this layer and the previous layers which connect forward to it.

        Parameters
        ---------------
        layer_in_list: List of tuples, of the form: layer_in_list: [(layer_name_1, delay_list_1),
                (layer_name_2, delay_list_2), etc.]. 
            Each element in layer_list is of the form (layer, delay_list) where layer is 
            a layer which connects forward to RNNLayer (of type RNNLayer or InLayer),
            and delay_list is a list of integers which denotes the list of delays between
            that layer and the current layer.
        
        Returns
        ----------
        None.          
        """        
        
        assert isinstance(layer_in_list, list), "In class %s: layer_in_list should be a list" % (self.__class__.__name__)
        for (layer, delay_list) in layer_in_list:
            assert isinstance(layer, (InLayer, RNNLayer)), "In class %s: layer should be either InLayer or RNNLayer" % (self.__class__.__name__)
            assert isinstance(delay_list, list), "In class %s: Delay list of connection from %s to %s should be a list" % (layer_name, self.name, self.__class__.__name__)

            layer_name = layer.get_name()

            self.builder.all_connections_dict[self.name, layer_name] = delay_list
            if layer_name in self.builder.connect_to_dict:
                self.builder.connect_to_dict[layer_name].add(self.name)
            else:
                self.builder.connect_to_dict[layer_name] = set()
                self.builder.connect_to_dict[layer_name].add(self.name)
                
            if self.name in self.builder.connect_back_dict:
                self.builder.connect_back_dict[self.name].add(layer_name)
            else:
                self.builder.connect_back_dict[self.name] = set()
                self.builder.connect_back_dict[self.name].add(layer_name)


class RNNBuilder:
    """ Base class to build Recurrent Neural Network (RNN). This class handles the building 
    of the computational graph as well as training and testing procedures for the RNN. 
    Training is done by using the Backpropagation through time (BPTT) algorithm. 
    Implementation is based on the details shown in [1] (chapter 14). Arbitrary recurrent
    connections is supported. To build a RNN, first inherit this class, then define the layers
    in the __init__() method, and define the connections between those layers in the define()
    method. An example (several other examples can be seen in rnn_test.py):
    
    class OneSimpleRNN(RNNBuilder):
        def __init__(self, name):
            super(self.__class__, self).__init__(name)
            n_in = 2
            self.input_layer = InLayer(n_in, self, name="input_1", act_func="linear")
            self.layer_1 = RNNLayer(3, self,  name="layer_1", act_func="sigmoid")
            self.layer_2 = RNNLayer(1, self,  name="layer_2", act_func="relu", is_output=True)
    
        def define(self):
            self.layer_1([
                (self.input_layer, [0]),
                (self.layer_2, [1]), 
            ])
            self.layer_2([
                (self.layer_1, [0]), 
            ])        
    
    in which InLayer denotes the layers used as input, and is_output=True denotes that
    the layer will be used as part of the output. In the define() function, each layer acceps 
    a list of tuple, the first element of which is one of the previous layers, and the second
    element is the list of delays (0 denotes that it's a normal connection, 1 denotes that 
    h(t) is turned into h(t-1), etc.).
    
    [1] M. Hagan et al., Neural network design (2nd ed.), 2014. 
    
    Methods (basically I tried to expose a Keras-like API)
    -----------
    define()
        Define the network connections (supports arbitrary recurrent connnections).
    compile()
        Builds the computational graph for the network.
    forward(input_dict)
        Performs the forward pass based on the input. 
    backward(loss_grad_dict, input_dict)
        Performs the backward pass by using the Backpropagation through time (BPTT) algorithm.
    zero_grad()
        Zero out all the gradients of the RNN.
    summary()
        Prints the summary of the RNN. 
    """

    def __init__(self, name):
        self.name = name
        self.layer_counter = 0
        
        # This variable help check whether you have called the compile() method
        # before training and running the network. 
        self.called_compile = False 
        
        self.layer_dict = {}
        self.iw = {} # Input weights
        self.lw = {} # Layer weights
        self.bias = {} # Layer bias
        self.iw_grad = {} # Gradient of input weights        
        self.lw_grad = {} # Gradient of layer weights
        self.bias_grad = {} # Gradient of layer bias
        self.out = {} # Outputs of each layer
        
        # Proper inputs and outputs (those defined explicitly)
        self.proper_input_layer_list = []
        self.proper_output_layer_list = []
 
        # Input and output layers (defined according to the textbook)
        self.input_layer_set = set() # Denoted as X in the textbook
        # self.output_layer_set = set() # Denoted as U in the textbook
 
        self.output_layer_set = [] # Denoted as U in the textbook
 
        # Simulation and backpropagation order (defined according to the textbook)
        self.simulation_order = []
        self.backprop_order = []
        
        # This part stores all connections and their delays
        self.all_connections_dict = {}
        
        # This variable stores all layers which a layer connects to
        self.connect_to_dict = {}
        
        # This variable stores all layers which a layer connects backward to
        self.connect_back_dict = {}
        
        # This part is defined according to the textbook
        # self.input_to_layer_dict[layer_m] denotes the list of input vectors that connect to layer m (I_m in the textbook).        
        self.input_to_layer_dict = {} 
        # self.layer_forward_dict[layer_m] denotes the list of layers that directly connect forward to layer m (L^f_m in the textbook).
        self.layer_forward_dict = {}
        # self.layer_backward_no_delay_dict[layer_m] denotes the list of layers that are directly connected backwards
        # to layer m (or to which layer m connects forward) and that contains no delays in the connection (L^b_m in the textbook).
        self.layer_backward_no_delay_dict = {} 
        
        # Delay from input-to-layer (DI_{m,l} in the textbook) and from layer-to-layer (DL_{m,l} in the textbook)         
        self.delay_in_dict = {}
        self.delay_layer_dict = {}
        
        # Layers that have connections from output layers (E^U_LW in the textbook)
        self.layer_connect_from_output_layer_dict = {}
        
        # Layers that connect to input layers (E^X_LW in the textbook)
        self.layer_connect_to_input_layer_dict = {}

   
    def define(self):
        """ Defines the connections in the RNN (supports arbitrary recurrent connections).
        This function must be implemented in a new class which inherits from RNNBuilder.
        In this functions, the connections between the layers, defined in __init__ of that class, 
        has to be defined. 

        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
        
        raise NotImplementedError

    
    def compile(self, optimizer_name='adam', loss_name='cross_entropy'):   
        """ Builds the computational graph for the network.
        This function must be called before running the network.

        Parameters
        ---------------
        optimizer_name: String.
            The optimizer to use. Default is 'adam'.
        loss_name: String.
            The loss to use. Default is 'cross_entropy'.
        
        Returns
        ----------
        None.
        """
        
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.called_compile = True    
        self.backprop_order = self.simulation_order[::-1]
        
        # Initialize out layers 
        for layer_name, layer in self.layer_dict.items():            
            if isinstance(self.layer_dict[layer_name], InLayer):
                continue
            if self.layer_dict[layer_name].is_output is True:
                self.proper_output_layer_list.append(layer_name)
  
        for layer_name, layer in self.layer_dict.items():
            try:
                layer_dest_list = self.connect_to_dict[layer_name]
            except:
                self.connect_to_dict[layer_name] = set()
        
        # Find input layers
        for layer_name, layer in self.layer_dict.items():
            if isinstance(self.layer_dict[layer_name], InLayer):
                continue

            connect_back_list = self.connect_back_dict[layer_name]
            for layer_back_name in connect_back_list:
                if isinstance(self.layer_dict[layer_back_name], InLayer):
                    self.input_layer_set.add(layer_name)
                    continue
            layer_dest_list = self.connect_to_dict[layer_name]
                         
            for layer_dest_name in layer_dest_list:
                delay_list = self.all_connections_dict[layer_dest_name, layer_name] 
                if (len(delay_list) == 1) and (delay_list[0] == 0):
                    continue
                else:
                    self.input_layer_set.add(layer_dest_name)
        
        # Find output layers
        for layer_name, layer in self.layer_dict.items():
            if isinstance(self.layer_dict[layer_name], InLayer):
                continue
                
            if layer.is_output is True:
                # self.output_layer_set.add(layer_name)
                self.output_layer_set.append(layer_name)
                continue
                
            layer_dest_list = self.connect_to_dict[layer_name]
            for layer_dest_name in layer_dest_list:
                delay_list = self.all_connections_dict[layer_dest_name, layer_name] 
                if (len(delay_list) == 1) and (delay_list[0] == 0):
                    continue
                else:
                    # self.output_layer_set.add(layer_name)
                    self.output_layer_set.append(layer_name)
                    continue
        
        # Find the set of indices of input vectors that connect to layer m
        for layer_name, layer in self.layer_dict.items():
            if isinstance(self.layer_dict[layer_name], InLayer):
                continue
            self.input_to_layer_dict[layer_name] = set()
                
        for layer_name, layer in self.layer_dict.items():
            if isinstance(self.layer_dict[layer_name], InLayer):
                layer_dest_list = self.connect_to_dict[layer_name]
                for layer_dest_name in layer_dest_list:
                    self.input_to_layer_dict[layer_dest_name].add(layer_name)
        
        # Find the input delays (DI_{m,l} in the textbook)
        for (layer_dest_name, layer_name) in self.all_connections_dict:
            if isinstance(self.layer_dict[layer_name], InLayer):
                self.delay_in_dict[layer_dest_name, layer_name] = self.all_connections_dict[layer_dest_name, layer_name]
        
        # Find the delays between layers (DL_{m,l} in the textbook)
        for (layer_dest_name, layer_name) in self.all_connections_dict:
            if isinstance(self.layer_dict[layer_name], InLayer) is False:
                self.delay_layer_dict[layer_dest_name, layer_name] = self.all_connections_dict[layer_dest_name, layer_name]

        # Find forward connections (L^f_m in the textbook)
        for layer_name, layer in self.layer_dict.items():
            if isinstance(self.layer_dict[layer_name], InLayer):
                continue
            self.layer_forward_dict[layer_name] = []
            connect_back_list = self.connect_back_dict[layer_name]
            for layer_back_name in connect_back_list:
                if isinstance(self.layer_dict[layer_back_name], InLayer):
                    continue
                self.layer_forward_dict[layer_name].append(layer_back_name)
        
        # Find backward connections with no delays (L^b_m in the textbook)
        for layer_name, layer in self.layer_dict.items():
            if isinstance(self.layer_dict[layer_name], InLayer):
                continue
            self.layer_backward_no_delay_dict[layer_name] = []
            connect_to_list = self.connect_to_dict[layer_name]
            for layer_dest_name in connect_to_list:
                if isinstance(self.layer_dict[layer_dest_name], InLayer):
                    continue
                # No delay
                delay_list = self.all_connections_dict[layer_dest_name, layer_name]
                if (len(delay_list) == 1) and (delay_list[0] == 0):
                    self.layer_backward_no_delay_dict[layer_name].append(layer_dest_name)
        
        # Find layers that have connections from output layers (E^U_LW in the textbook)
        for layer_name, layer in self.layer_dict.items():
            if isinstance(self.layer_dict[layer_name], InLayer):
                continue
            self.layer_connect_from_output_layer_dict[layer_name] = []
            connect_back_list = self.connect_back_dict[layer_name]
            for layer_back_name in connect_back_list:
                if layer_back_name not in self.output_layer_set:
                    continue
                delay_list = self.all_connections_dict[layer_name, layer_back_name]
                # Contains delays
                if (len(delay_list) == 1) and (delay_list[0] == 0):
                    continue
                self.layer_connect_from_output_layer_dict[layer_name].append(layer_back_name)    
                
        # Find layers that connect to input layers (E^X_LW in the textbook)
        for layer_name, layer in self.layer_dict.items():
            if isinstance(self.layer_dict[layer_name], InLayer):
                continue
            self.layer_connect_to_input_layer_dict[layer_name] = []
            connect_to_list = self.connect_to_dict[layer_name]
            for layer_dest_name in connect_to_list:
                if layer_dest_name not in self.input_layer_set:
                    continue
                delay_list = self.all_connections_dict[layer_dest_name, layer_name]
                # Contains delays
                if (len(delay_list) == 1) and (delay_list[0] == 0):
                    continue
                self.layer_connect_to_input_layer_dict[layer_name].append(layer_dest_name)
        
        # Initialize weights matrices
        for (layer_dest_name, layer_name), delay_list in self.all_connections_dict.items():
            n_delay = len(delay_list)
            n_in = self.layer_dict[layer_name].n_dim
            n_out = self.layer_dict[layer_dest_name].n_dim
            if isinstance(self.layer_dict[layer_name], InLayer):                
                self.iw[(layer_dest_name, layer_name)] = np.zeros((n_out, n_in, n_delay))
                for delay_idx in range(n_delay):
                    self.iw[(layer_dest_name, layer_name)][:, :, delay_idx] = init_matrix(n_in, n_out, method="normal")                
            else:
                self.lw[(layer_dest_name, layer_name)] = np.zeros((n_out, n_in, n_delay))
                for delay_idx in range(n_delay):
                    self.lw[(layer_dest_name, layer_name)][:, :, delay_idx] = init_matrix(n_in, n_out, method="normal")

        # Initialize bias
        for layer_name, layer in self.layer_dict.items():
            if isinstance(self.layer_dict[layer_name], InLayer):
                continue
            n_dim = layer.n_dim
            self.bias[layer_name] = np.random.normal(size=(n_dim))
            # self.bias[layer_name] = init_matrix(1, n_dim, method="normal")
        
        # Initialize weight and bias gradients
        for (layer_dest_name, layer_name) in self.iw:
            self.iw_grad[(layer_dest_name, layer_name)] = np.zeros_like(self.iw[(layer_dest_name, layer_name)])
        for (layer_dest_name, layer_name) in self.lw:
            self.lw_grad[(layer_dest_name, layer_name)] = np.zeros_like(self.lw[(layer_dest_name, layer_name)])
        for layer_name in self.bias:
            self.bias_grad[layer_name] = np.zeros_like(self.bias[layer_name])

        self._begin_track_shapes()


    def _begin_track_shapes(self):
        """ Begin tracking the shapes of the weights, bias and gradient arrays of the RNN.
        This function is used for debugging purposes. 
        
        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
    
        self.iw_shape = {}
        self.lw_shape = {}
        self.bias_shape = {}
        self.iw_grad_shape = {}
        self.lw_grad_shape = {}
        self.bias_grad_shape = {}
                
        for key in self.iw:
            self.iw_shape[key] = self.iw[key].shape            
        for key in self.lw:
            self.lw_shape[key] = self.lw[key].shape            
        for key in self.bias:
            self.bias_shape[key] = self.bias[key].shape

        for key in self.iw_grad:
            self.iw_grad_shape[key] = self.iw_grad[key].shape            
        for key in self.lw_grad:
            self.lw_grad_shape[key] = self.lw_grad[key].shape            
        for key in self.bias_grad:
            self.bias_grad_shape[key] = self.bias_grad[key].shape


    def _check_shapes(self):
        """ Check if the shapes of the weights and bias as well as their gradients 
        has changed or not, which usually indicates that something is wrong.
        This function is used for debugging purposes. Apply this function as a 
        decorator in relevant methods in order to catch implementation errors. 
        
        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
    
        # Check for differences between the keys of the weights and gradients
        assert self.iw.keys() == self.iw_grad.keys(), \
            "In class %s: The keys of input-to-layer weights (%s) and its gradients (%s) are different." % (self.__class__.__name__, self.iw.keys(), self.iw_grad.keys())
        assert self.lw.keys() == self.lw_grad.keys(), \
            "In class %s: The keys of layer-to-layer weights (%s) and its gradients (%s) are different." % (self.__class__.__name__, self.lw.keys(), self.lw_grad.keys())
        assert self.bias.keys() == self.bias_grad.keys(), \
            "In class %s: The keys of the bias (%s) and its gradients (%s) are different." % (self.__class__.__name__, self.bias.keys(), self.bias_grad.keys())
    
        # Check the shapes of weights and bias
        for key in self.iw:
            assert isinstance(self.iw[key],np.ndarray), \
                "In class %s: Expected type of input-to-layer of layers: %s is np.ndarray, instead got %s" % (self.__class__.__name__, key, type(self.iw[key])) 
            assert self.iw[key].shape == self.iw_shape[key], \
                "In class %s: The shapes of input-to-layer weights of layers: %s have been changed from %s to %s." % (self.__class__.__name__, key, self.iw_shape[key], self.iw[key].shape)
        
        for key in self.lw:
            assert isinstance(self.lw[key],np.ndarray), \
                "In class %s: Expected type of layer-to-layer of layers: %s is np.ndarray, instead got %s" % (self.__class__.__name__, key, type(self.lw[key])) 
            assert self.lw[key].shape == self.lw_shape[key], \
                "In class %s: The shapes of layer-to-layer weights of layers: %s have been changed from %s to %s." % (self.__class__.__name__, key, self.lw_shape[key], self.lw[key].shape)

        for key in self.bias:
            assert isinstance(self.bias[key],np.ndarray), \
                "In class %s: Expected type of bias is np.ndarray, instead got %s" % (self.__class__.__name__, type(self.bias[key])) 
            assert self.bias[key].shape == self.bias_shape[key], \
                "In class %s: The shapes of bias of layers: %s have been changed from %s to %s." % (self.__class__.__name__, key, self.bias_shape[key], self.bias[key].shape)

        # Check the shapes of the gradient of weights and bias
        for key in self.iw_grad:
            assert isinstance(self.iw_grad[key],np.ndarray), \
                "In class %s: Expected type of gradient of input-to-layer of layers: %s is np.ndarray, instead got %s" % (self.__class__.__name__, key, type(self.iw_grad[key])) 
            assert self.iw_grad[key].shape == self.iw_grad_shape[key], \
                "In class %s: The shapes of the gradient of input-to-layer weights of layers: %s have been changed from %s to %s." % (self.__class__.__name__, key, self.iw_grad_shape[key], self.iw_grad[key].shape)

        for key in self.lw_grad:
            assert isinstance(self.lw_grad[key],np.ndarray), \
                "In class %s: Expected type of gradient of layer-to-layer of layers: %s is np.ndarray, instead got %s" % (self.__class__.__name__, key, type(self.lw_grad[key])) 
            # set_trace()
            assert self.lw_grad[key].shape == self.lw_grad_shape[key], \
                "In class %s: The shapes of gradient of layer-to-layer weights of layers: %s have been changed from %s to %s." % (self.__class__.__name__, key, self.lw_grad_shape[key], self.lw_grad[key].shape)

        for key in self.bias_grad:
            assert isinstance(self.bias_grad[key],np.ndarray), \
                "In class %s: Expected type of gradient of bias is np.ndarray, instead got %s" % (self.__class__.__name__, type(self.bias_grad[key])) 
            assert self.bias_grad[key].shape == self.bias_grad_shape[key], \
                "In class %s: The shapes of the gradient of bias of layers: %s have been changed from %s to %s." % (self.__class__.__name__, key, self.bias_grad_shape[key], self.bias_grad[key].shape)

   
    @check_weights_and_gradient_shapes
    def forward(self, input_dict):
        """ Performs a forward pass.

        Parameters
        ---------------
        input_dict: Dictionary of numpy arrays. 
            The input, expressed as a Python dictionary to support multiple inputs such as in BidirectionalRNN.
            input_dict[layer_m_name] denotes the input from layer_m (which must be an InLayer), and has 
            size (batch_size, n_step, n_features) where batch_size is the number of inputs in the batch,
            n_step is the number of timesteps, and n_features is the number of input features.
        
        Returns
        ----------
        pred_dict: Dictionary of numpy arrays.
            The output prediction, with one element for each layer in which is_output is set to True in the 
            __init__() method. The output of layer m is of size (batch_size, n_step, layer_m_dim).          
        """

        if self.called_compile is False:
            raise Exception("In class %s: compile() must be called before running the network" % (self.__class__.__name__))
        
        for key in input_dict:
            inp = input_dict[key]
            (batch_size, n_step, n_features) = inp.shape
            break

        # Initialize output
        for layer_name, layer in self.layer_dict.items():
            # In this case we also include the inputs for convenience
            n_dim = layer.n_dim
            self.out[layer_name] = np.zeros((batch_size, n_step, n_dim))
            
        # Perform the forward pass
        for step_idx in range(n_step):
            logging.debug("Step: %d" % (step_idx))
            for layer_name in self.layer_dict:
                if isinstance(self.layer_dict[layer_name], InLayer):
                    continue
                
                layer_prev_list = self.layer_forward_dict[layer_name]
                layer_input_prev_list = self.input_to_layer_dict[layer_name]
                
                # Sum from previous layers
                for layer_prev in layer_prev_list:                    
                    # logging.debug(layer_prev)
                    delay_list = self.delay_layer_dict[(layer_name, layer_prev)]                      
                    for delay_idx, delay in enumerate(delay_list):                        
                        term1_left = self.lw[(layer_name, layer_prev)][:, :, delay_idx]
                        if step_idx-delay < 0:
                            term1_right = np.zeros_like(self.out[layer_prev][:, 0, :])
                        else:
                            term1_right = self.out[layer_prev][:, int(step_idx-delay), :]
                            # set_trace()
                        term1 = np.einsum("ij,nj->ni", term1_left, term1_right)
                        logging.debug("term1_left:")
                        logging.debug(term1_left)
                        logging.debug("term1_right:")
                        logging.debug(term1_right)
                        logging.debug("term1:")
                        logging.debug(term1)
                        self.out[layer_name][:, step_idx, :] += term1
                        # set_trace()
                logging.debug("Summed from previous layers")
                # set_trace()
                        
                # Sum from inputs
                for layer_input_prev in layer_input_prev_list:
                    logging.debug(layer_input_prev)
                    delay_list = self.delay_in_dict[(layer_name, layer_input_prev)]
                    for delay_idx, delay in enumerate(delay_list):
                        term2_left = self.iw[(layer_name, layer_input_prev)][:, :, delay_idx]
                        if step_idx-delay < 0:
                            term2_right = np.zeros_like(input_dict[layer_input_prev])[:, 0, :]
                        else:
                            term2_right = input_dict[layer_input_prev][:, int(step_idx-delay), :]
                        term2 = np.einsum("ij,nj->ni", term2_left, term2_right)
                        logging.debug("term2:")
                        logging.debug(term2)
                        self.out[layer_name][:, step_idx, :] += term2
                logging.debug("Summed from inputs")

                # Sum with bias
                logging.debug("Sum with bias")
                logging.debug("Bias:")
                logging.debug(self.bias[layer_name])
                self.out[layer_name][:, step_idx, :] += self.bias[layer_name]
                logging.debug("Output is now:")
                logging.debug(self.out[layer_name][:, step_idx, :])
                
                # Activation
                logging.debug("Activation")
                self.out[layer_name][:, step_idx, :] = eval_activation_func(self.out[layer_name][:, step_idx, :], self.layer_dict[layer_name].act_func)
                logging.debug("Output is now:")
                logging.debug(self.out[layer_name][:, step_idx, :])
        
        # Return output
        pred_dict = {}
        for layer_name in self.proper_output_layer_list:
            pred_dict[layer_name] = deepcopy(self.out[layer_name])

        return pred_dict

        
    def fit(self, train_dataloader, val_dataloader=None, epochs=100,
            steps_per_epoch=None, val_steps_per_epoch=None,
            optimizer_params=None, loss_weights=None, inputs=None, outputs=None):
        if self.optimizer_name == "adam":
            optimizer = AdamOptimizerRNN(optimizer_params) 

        train_loss_list = []
        val_loss_list = []
        for epoch_idx in range(epochs):
            avg_train_loss_dict = None
            avg_val_loss_dict = None
            self.zero_grad()
            for step_idx in range(steps_per_epoch):
                (train_data_batch_orig, train_labels_batch_orig) = train_dataloader.next()
                train_data_batch = {}
                train_labels_batch = {}
                for input_idx, input_entry in enumerate(inputs):
                    train_data_batch[input_entry] = train_data_batch_orig[input_idx]
                for output_idx, output_entry in enumerate(outputs):
                    train_labels_batch[output_entry] = train_labels_batch_orig[output_idx]
#                (train_data_batch, train_labels_batch) = train_dataloader.next()
                out_train = self.forward(train_data_batch)

                train_loss_dict, train_loss_grad_dict = eval_loss_func_rnn(train_labels_batch, out_train, loss_func=self.loss_name)
                if avg_train_loss_dict is None:
                    avg_train_loss_dict = train_loss_dict
                else:
                    for key1 in train_loss_dict:
                        avg_train_loss_dict[key1] += train_loss_dict[key1]
                self.backward(train_loss_grad_dict, train_data_batch)
                optimizer.step(epoch_idx, self)
                self.zero_grad()
            
            for key1 in avg_train_loss_dict:
                avg_train_loss_dict[key1] /= (1.0*steps_per_epoch)

            avg_train_loss = 0.0
            for key1 in avg_train_loss_dict:
                avg_train_loss += loss_weights[key1]*avg_train_loss_dict[key1]

            train_loss_list.append(avg_train_loss)

            if val_dataloader is not None:
                for step_idx in range(val_steps_per_epoch):
                    (val_data_batch_orig, val_labels_batch_orig) = val_dataloader.next()
#                    (train_data_batch_orig, train_labels_batch_orig) = train_dataloader.next()
                    val_data_batch = {}
                    val_labels_batch = {}
                    for input_idx, input_entry in enumerate(inputs):
                        val_data_batch[input_entry] = val_data_batch_orig[input_idx]
                    for output_idx, output_entry in enumerate(outputs):
                        val_labels_batch[output_entry] = val_labels_batch_orig[output_idx]
#                    (val_data_batch, val_labels_batch) = val_dataloader.next()
                    out_val = self.forward(val_data_batch)
                    val_loss_dict, val_loss_grad_dict = eval_loss_func_rnn(val_labels_batch, out_val, loss_func=self.loss_name)
                    if avg_val_loss_dict is None:
                        avg_val_loss_dict = val_loss_dict
                    else:
                        for key1 in val_loss_dict:
                            avg_val_loss_dict[key1] += val_loss_dict[key1]

                for key1 in avg_val_loss_dict:
                    avg_val_loss_dict[key1] /= (1.0*val_steps_per_epoch)

                avg_val_loss = 0.0
                for key1 in avg_val_loss_dict:
                    avg_val_loss += loss_weights[key1]*avg_val_loss_dict[key1]
                print("Epoch %d, train loss: %f, validation loss: %f" % (epoch_idx+1, avg_train_loss, avg_val_loss))
                val_loss_list.append(avg_val_loss)
            else:
                print("Epoch %d, train loss: %f" % (epoch_idx+1, avg_train_loss))
                
        return (train_loss_list, val_loss_list)


    def evaluate(self, dataloader, steps_per_epoch=None, loss_weights=None, inputs=None, outputs=None):
        avg_loss_dict = None
        
        y_pred_all = []
        y_true_all = []
        
        for step_idx in range(steps_per_epoch):
#            (data_batch, labels_batch) = dataloader.next()
            (data_batch_orig, labels_batch_orig) = dataloader.next()
            data_batch = {}
            labels_batch = {}
            for input_idx, input_entry in enumerate(inputs):
                data_batch[input_entry] = data_batch_orig[input_idx]
            for output_idx, output_entry in enumerate(outputs):
                labels_batch[output_entry] = labels_batch_orig[output_idx]
#            (val_data_batch, val_labels_batch) = val_dataloader.next()
            out = self.forward(data_batch)
            
            # This part is a little bit hacky. TODO: Rewrite it. 
            # set_trace()
            y_pred_all.append(out[outputs[0]][0, -1, 0])
            y_true_all.append(labels_batch_orig[0][0, -1, 0])            
            
            loss_dict, loss_grad_dict = eval_loss_func_rnn(labels_batch, out, loss_func=self.loss_name)
            if avg_loss_dict is None:
                avg_loss_dict = loss_dict
            else:
                for key1 in loss_dict:
                    try:
                        avg_loss_dict[key1] += loss_dict[key1]
                    except:
                        set_trace() 

        for key1 in avg_loss_dict:
            avg_loss_dict[key1] /= (1.0*steps_per_epoch)

        avg_loss = 0.0
        for key1 in avg_loss_dict:
            avg_loss += loss_weights[key1]*avg_loss_dict[key1]

        return avg_loss, y_pred_all, y_true_all
        

    def backward(self, loss_grad_dict, input_dict):
        """ Performs a backward pass by using the Backpropagation through time (BPTT) algorithm.

        Parameters
        ---------------
        loss_grad_dict: Dictionary of numpy arrays. 
            The loss gradient (explicit derivatives) of the layers used as output. Each element is of size
            (batch_size, n_step, n_out_dim).            
        input_dict: Dictionary of numpy arrays. 
            The input, expressed as a Python dictionary to support multiple inputs such as in BidirectionalRNN.
            input_dict[layer_m_name] denotes the input from layer_m (which must be an InLayer), and has 
            size (batch_size, n_step, n_features) where batch_size is the number of inputs in the batch,
            n_step is the number of timesteps, and n_features is the number of input features.
        
        Returns
        ----------
        None.          
        """
        
        if self.called_compile is False:
            raise Exception("In class %s: compile() must be called before running the network" % (self.__class__.__name__))
        
        logging.debug("Begin BPTT algorithm")
        # Get the dimensions
        for key in loss_grad_dict:
            (batch_size, n_step, n_out_dim) = loss_grad_dict[key].shape
            break
#        set_trace()
        
        # Sensititvity matrices 
        # ss[layer_u, layer_m] has size (batch_size, layer_u_dim, layer_m_dim, n_step)
        ss_dict = {}
        
        # The derivatives with respect to the output layers
        # div_F_wrt_outputs[layer_u] is of size (batch_size, n_step, layer_u_dim))
        div_F_wrt_outputs = {}
        for layer_u  in self.backprop_order:
            if isinstance(self.layer_dict[layer_u], InLayer):
                continue
            if layer_u not in self.output_layer_set:
                continue
            layer_u_dim = layer_u_dim = self.layer_dict[layer_u].n_dim
            div_F_wrt_outputs[layer_u] = np.zeros((batch_size, n_step, layer_u_dim))
        
        # Defined by Equations (14.48), (14.51) and (14.52) in the textbook.
        # I just don't know what to call it
        # d_dict[layer_m] is of size (batch_size, n_step, layer_m_dim)
        d_dict = {}
        for layer_m in self.layer_dict:
            if isinstance(self.layer_dict[layer_m], InLayer):
                continue
            layer_m_dim = self.layer_dict[layer_m].n_dim
            d_dict[layer_m] = np.zeros((batch_size, n_step, layer_m_dim))
      
        for step_idx in reversed(range(n_step)):
            logging.debug("Step: %d" % (step_idx))
            logging.debug("Begin calculating the sensitivity matrices")
            
            U1 = []
            E_S = {}
            E_U_S = {} # E^U_S[x] denotes the set of output layers which have a nonzero sensitivity with input layer x (static connection)
        
            # for layer_u in self.output_layer_set:
            for layer_u in self.layer_dict:
                E_S[layer_u] = []
                E_U_S[layer_u] = []

            for layer_m in self.backprop_order:
                if isinstance(self.layer_dict[layer_m], InLayer):
                    continue
                logging.debug("Layer m: %s" % (layer_m))
                for layer_u in reversed(U1):
                    E_S_layer_u = E_S[layer_u]
                    l_list = [elem for elem in E_S_layer_u if elem in self.layer_backward_no_delay_dict[layer_m]]                    
                    if (len(l_list) == 0) or (l_list is None):
                        continue
                  
                    logging.debug("S^{%s,%s}(%d) =" % (layer_u, layer_m, step_idx)) 
                    logging.debug("At layer %s, layer_l_list is:" % (layer_m))
                    logging.debug(l_list)
                    logging.debug(E_S_layer_u)
                    logging.debug(self.layer_backward_no_delay_dict[layer_m])
         
                    layer_u_dim = self.layer_dict[layer_u].n_dim
                    layer_m_dim = self.layer_dict[layer_m].n_dim
                    term_left = np.zeros((batch_size, layer_u_dim, layer_m_dim))
         
                    for layer_l in  reversed(l_list):
                        logging.debug("++ Layer m: %s. Layer l: %s" % (layer_m, layer_l))
                        if (layer_u, layer_m) not in ss_dict:
                            layer_u_dim = self.layer_dict[layer_u].n_dim
                            layer_m_dim = self.layer_dict[layer_m].n_dim
                            ss_dict[(layer_u, layer_m)] = np.zeros((batch_size, layer_u_dim, layer_m_dim, n_step))
                        logging.debug(" -- S^{%s,%s}(%d)*LW^{%s, %s}(0)" % (layer_u, layer_l, step_idx, layer_l, layer_m))                        
                        try:
                            term_left += np.einsum(
                                "nul,lm->num",
                                ss_dict[(layer_u, layer_l)][:, :, :, step_idx],
                                self.lw[(layer_l, layer_m)][:, :, 0],
                            )
                        except:
                            print("term_left is wrong here")
                            set_trace()

                    act_func = self.layer_dict[layer_m].act_func
                    out_grad = eval_activation_func_gradient(self.out[layer_m][:, step_idx, :], act_func)
                    logging.debug(" -- * Fdot^{%s}(n^{%s}(%d))" % (layer_m, layer_m, step_idx))
                    ss_dict[(layer_u, layer_m)][:, :, :, step_idx] =  np.einsum(
                        "num,nm->num",
                        term_left,
                        out_grad,
                    )      
                    
                    E_S[layer_u].append(layer_m)                                            
                    # if layer_m in self.output_layer_set:
                    E_U_S[layer_m].append(layer_u)
                
                if layer_m in self.output_layer_set:
                    logging.debug("Layer m (%s) is in output set: Add m to the set U', E_S(m) and E^U_S(m)" % (layer_m))
                    if (layer_m, layer_m) not in ss_dict:
                        layer_m_dim = self.layer_dict[layer_m].n_dim
                        ss_dict[(layer_m, layer_m)] = np.zeros((batch_size, layer_m_dim, layer_m_dim, n_step))                    
                    logging.debug("S^{%s,%s}(%d)=Fdot^{%s}(n^{%s}(%d))" % (layer_m, layer_m, step_idx, layer_m, layer_m, step_idx))
                    act_func = self.layer_dict[layer_m].act_func
                    temp = eval_activation_func_gradient(self.out[layer_m][:, step_idx, :], act_func)
                    ss_dict[(layer_m, layer_m)][:, :, :, step_idx] = np.array([np.diag(row) for row in temp])

                    U1.append(layer_m)
                    if layer_m in E_S:
                        E_S[layer_m].append(layer_m)
                    else:
                        E_S[layer_m] = [layer_m]
                    if layer_m in E_U_S:
                        E_U_S[layer_m].append(layer_m)
                    else:
                        E_U_S[layer_m] = [layer_m]
                    
                logging.debug("At the end of the loop (w.r.t layer m (%s))" % (layer_m))
                logging.debug("U':")
                logging.debug(U1)                
                logging.debug("E_S(m) (m=%s):" % (layer_m))
                if layer_m in E_S:
                    logging.debug(E_S[layer_m])
                else:
                    logging.debug("[]")
                logging.debug("E^U_S(m)':")
                if layer_m in E_U_S:
                    logging.debug(E_U_S[layer_m])
                else:
                    logging.debug("[]")
                    
                logging.debug("E_S:")
                logging.debug(E_S)
                logging.debug("E^U_S")
                logging.debug(E_U_S)
            
            logging.debug("Finished calculating the sensitivity matrices")
            
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("----------------------------------------------------------------")
            logging.debug("Begin calculating the derivatives w.r.t the outputs")
            logging.debug("----------------------------------------------------------------")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            
            for layer_u  in self.backprop_order:
                if isinstance(self.layer_dict[layer_u], InLayer):
                    continue
                if layer_u not in self.output_layer_set:
                    continue
                logging.debug(layer_u)
                logging.debug("div_F w.r.t %s = " % (layer_u))
                
                layer_u_dim = self.layer_dict[layer_u].n_dim
                                
                for layer_x in self.layer_connect_to_input_layer_dict[layer_u]:                    
                    layer_x_dim = self.layer_dict[layer_x].n_dim
                    delay_list = self.all_connections_dict[(layer_x, layer_u)]
                    for delay_idx, delay in enumerate(delay_list):
                        logging.debug(" -- LW^{%s, %s}(%d)" % (layer_x, layer_u, delay))
                        ss_mult_div_F_t_plus_d = None
                        for layer_u1 in E_U_S[layer_x]:
                            logging.debug(" ----- S^{%s, %s}(t+%d) x divF^{%s} (t+%d)" % (layer_u1, layer_x, delay, layer_u1, delay))
                            if step_idx+delay<=(n_step-1):
                                div_F_t_plus_d = div_F_wrt_outputs[layer_u1][:, step_idx+delay, :]
                                ss_u1_x_t_plus_d = ss_dict[(layer_u1, layer_x)][:, :, :, step_idx+delay]
                            else:
                                div_F_t_plus_d = np.zeros_like(div_F_wrt_outputs[layer_u1][:, 0, :])
                                ss_u1_x_t_plus_d = np.zeros_like(ss_dict[(layer_u1, layer_x)][:, :, :, 0])
                            if ss_mult_div_F_t_plus_d is None:
                                ss_mult_div_F_t_plus_d = np.einsum(
                                    "nij,ni->nj",
                                    ss_u1_x_t_plus_d,
                                    div_F_t_plus_d
                                )
                            else:
                                ss_mult_div_F_t_plus_d += np.einsum(
                                    "nij,ni->nj",
                                    ss_u1_x_t_plus_d,
                                    div_F_t_plus_d
                                )
                                logging.debug(ss_mult_div_F_t_plus_d)
                        if ss_mult_div_F_t_plus_d is None:
                            ss_mult_div_F_t_plus_d = np.zeros((batch_size, layer_x_dim))
                        lw_mult_ss_mult_div_F = np.einsum(
                            "ij,ni->nj",
                            self.lw[(layer_x, layer_u)][:, :, delay_idx],
                            ss_mult_div_F_t_plus_d,
                        )
                        div_F_wrt_outputs[layer_u][:, step_idx, :] += lw_mult_ss_mult_div_F
            
                if layer_u in loss_grad_dict:
                    div_F_wrt_outputs[layer_u][:, step_idx, :] += loss_grad_dict[layer_u][:, step_idx, :]
            
            # set_trace()
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("----------------------------------------------------------------")
            logging.debug("Begin calculating the multiplication between the sensitivities and the derivatives w.r.t. the outputs")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            logging.debug("")
            
            for layer_m in self.layer_dict:
                if isinstance(self.layer_dict[layer_m], InLayer):
                    continue
                layer_m_dim = self.layer_dict[layer_m].n_dim
                logging.debug("d^_{%s}(t) = " % (layer_m))
                for layer_u in E_U_S[layer_m]:
                    layer_u_dim = self.layer_dict[layer_u].n_dim
                    logging.debug("--- S^{%s, %s}(t) x divF_{a^{%s](t)}" % (layer_u, layer_m, layer_u))
                    d_dict[layer_m][:, step_idx, :] += np.einsum(
                        "nij,ni->nj",
                        ss_dict[layer_u, layer_m][:, :, :, step_idx],
                        div_F_wrt_outputs[layer_u][:, step_idx, :],
                    )
        
        # set_trace()
        # Compute gradients
        logging.debug("")
        logging.debug("")
        logging.debug("")
        logging.debug("----------------------------------------------------------------")
        logging.debug("Begin computing gradients")
        logging.debug("")
        logging.debug("")
        logging.debug("")
        
        for (layer_m, layer_l) in self.lw_grad:            
            layer_m_dim = self.layer_dict[layer_m].n_dim
            layer_l_dim = self.layer_dict[layer_l].n_dim
            delay_list = self.delay_layer_dict[(layer_m, layer_l) ]
            for delay_idx, delay in enumerate(delay_list):
                for step_idx in range(n_step):
                    if step_idx - delay >= 0:
                        # self.lw_grad[(layer_m, layer_l)][:, :, step_idx] = np.dot(np.transpose(d_dict[layer_m][:, step_idx, :]), self.out[layer_l][:, step_idx, :])
                        self.lw_grad[(layer_m, layer_l)][:, :, delay_idx] += np.einsum(
                            "nm,nl->ml",
                            d_dict[layer_m][:, step_idx, :],
                            self.out[layer_l][:, step_idx-delay, :],
                        )
                    else:
                        # In this case we do nothing since it would be just zeros
                        pass
        
        for (layer_m, layer_l) in self.iw_grad:
            layer_m_dim = self.layer_dict[layer_m].n_dim
            layer_l_dim = self.layer_dict[layer_l].n_dim
            delay_list = self.delay_in_dict[(layer_m, layer_l)]
            for delay_idx, delay in enumerate(delay_list):
                for step_idx in range(n_step):
                    if step_idx - delay >= 0:
#                        set_trace()
                        self.iw_grad[(layer_m, layer_l)][:, :, delay_idx] += np.einsum(
                            "nm,nl->ml",
                            d_dict[layer_m][:, step_idx, :],
                            input_dict[layer_l][:, step_idx-delay, :],
                        )
                    else:
                        # In this case we do nothing since it would be just zeros
                        pass
        
        for layer_m in self.layer_dict:
            if isinstance(self.layer_dict[layer_m], InLayer):
                continue
            layer_m_dim = self.layer_dict[layer_m].n_dim
            for step_idx in range(n_step):
                self.bias_grad[layer_m] += np.sum(d_dict[layer_m][:, step_idx], axis=0)


    @check_weights_and_gradient_shapes
    def zero_grad(self):
        """ Zero out all of the gradients of the RNN.

        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
        
        if self.called_compile is False:
            raise Exception("In class %s: compile() must be called before running the network" % (self.__class__.__name__))
            
        for (layer_m, layer_l) in self.lw_grad:
            self.lw_grad[(layer_m, layer_l)] = np.zeros_like(self.lw_grad[(layer_m, layer_l)])
            
        for (layer_m, layer_l) in self.iw_grad:
            self.iw_grad[(layer_m, layer_l)] = np.zeros_like(self.iw_grad[(layer_m, layer_l)])
            
        for layer_m in self.bias_grad:
            self.bias_grad[layer_m] = np.zeros_like(self.bias_grad[layer_m])
        pass

  
    @check_weights_and_gradient_shapes
    def summary(self):
        """ Print the general information of the RNN.

        Parameters
        ---------------
        None. 
        
        Returns
        ----------
        None.
        """
        
        if self.called_compile is False:
            raise Exception("In class %s: compile() must be called before running the network" % (self.__class__.__name__))
                
        print("")
        print("----- Begin RNN summary -----")
        print("Model name: %s" % (self.name))
        for layer_name in self.simulation_order:
            print("- Layer name: %s" % (layer_name))
            for layer_dest_name in self.connect_to_dict[layer_name]:                
                print("-- Connect to %s with delay list %s" % (layer_dest_name, self.all_connections_dict[layer_dest_name, layer_name]))                
        print("")

        print("Simulation order: %s" % (self.simulation_order))
        print("")
        
        print("Backpropagation order: %s" % (self.backprop_order))
        print("")
        
        print("Input layers: %s" % (self.input_layer_set))
        print("")
        
        print("Output layers: %s" % (self.output_layer_set))
        print("")
        
        for layer_name in self.simulation_order:
            try:
                self.input_to_layer_dict[layer_name]
            except:
                continue
            print("The set of indices of input vectors that connect to layer %s: %s" % (layer_name, self.input_to_layer_dict[layer_name]))
            print("")
        
        print("Input delays:")
        for ((layer_dest_name, layer_name), delay_list) in self.delay_in_dict.items():
            print("Input delays from %s to %s: %s" % (layer_name, layer_dest_name, delay_list))
        print("")
        
        print("Delays between layers:")
        for ((layer_dest_name, layer_name), delay_list) in self.delay_layer_dict.items():
            print("Delays from %s to %s: %s" % (layer_name, layer_dest_name, delay_list))
        print("")
        
        print("Forward connections:")
        for (layer_name, layer_list) in self.layer_forward_dict.items():
            print("Layers which connect forward to %s: %s" % (layer_name, layer_list))
        print("")
        
        print("Backward connections with no delays:")
        for (layer_name, layer_list) in self.layer_backward_no_delay_dict.items():
            print("Layers which connect backwards (with no delay) to %s: %s" % (layer_name, layer_list))
        print("")
        
        print("Layers that have connections from output layers:")
        for (layer_name, layer_list) in self.layer_connect_from_output_layer_dict.items():
            print("Layers which connect to %s that are output layers: %s" % (layer_name, layer_list))
        print("")
        
        print("Layers that connect to input layers:")
        for (layer_name, layer_list) in self.layer_connect_to_input_layer_dict.items():
            print("Layers which connect from %s that are input layers: %s" % (layer_name, layer_list))
        print("")
        
        print("Input-to-layer weights:")
        for (layer_dest_name, layer_name) in self.iw:
            print("Weight from input layer %s to layer %s has size: %s" % (layer_name, layer_dest_name, self.iw[(layer_dest_name, layer_name)].shape))
        print("")
        
        print("Layer-to-layer weights:")
        for (layer_dest_name, layer_name) in self.lw:
            print("Weight from layer %s to layer %s has size: %s" % (layer_name, layer_dest_name, self.lw[(layer_dest_name, layer_name)].shape))
        print("")
        
        print("----- End RNN summary -----")
        print("")
    

def main():
    pass

   
if __name__ == '__main__':
    main()
