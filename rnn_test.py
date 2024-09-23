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

from config import get_config

import warnings
warnings.filterwarnings('ignore')

# import tensorflow as tf

import logging
logging.basicConfig(level=get_config()["debug_mode"])

from rnn import InLayer, RNNLayer, RNNBuilder, check_weights_and_gradient_shapes, eval_loss_func_rnn
from utils import eval_activation_func, show_info


class MyRecurrentNeuralNetwork(RNNBuilder):
    def __init__(self, name):
        super(self.__class__, self).__init__(name)
        
        n_in = 5
        self.input_layer = InLayer(n_in, self, name="input_1")
        n_dim = 13
                
        self.layer_1 = RNNLayer(n_dim, self, name="layer_1")
        self.layer_2 = RNNLayer(n_dim, self, name="layer_2")
        self.layer_3 = RNNLayer(n_dim, self, name="layer_3", is_output=True)
        
    def define(self):
        # Figure 14.1 in the textbook
        self.layer_1([
            (self.input_layer, [0]),
            (self.layer_1, [1]),
            (self.layer_3, [1]),
        ])
        self.layer_2([
            (self.layer_1, [1]), 
            (self.layer_3, [1]),
        ])
        self.layer_3([
            (self.layer_2, [0]),
        ])
        

class SimpleRNN1(RNNBuilder):
    def __init__(self, name):
        super(self.__class__, self).__init__(name)
        
        n_in = 1
        self.input_layer = InLayer(n_in, self, name="input_1")
        n_dim = 1
        self.layer_1 = RNNLayer(n_dim, self,  name="layer_1", act_func="linear", is_output=True)
    
    def define(self):
        # Figure 14.2 in the textbook
        self.layer_1([
            (self.input_layer, [0, 1, 2]),
        ])
    
class SimpleRNN2(RNNBuilder):
    def __init__(self, name):
        super(self.__class__, self).__init__(name)        
        
        n_in = 1
        self.input_layer = InLayer(n_in, self, name="input_1")
        n_dim = 1
        self.layer_1 = RNNLayer(n_dim, self,  name="layer_1", act_func="linear", is_output=True)
        
    def define(self):
        # Figure 14.4 in the textbook
        self.layer_1([
            (self.input_layer, [0]),
            (self.layer_1, [1]),
        ])    

class SimpleRNN3(RNNBuilder):
    def __init__(self, name):
        super(self.__class__, self).__init__(name)        
        
        n_in = 3
        self.input_layer = InLayer(n_in, self, name="input_1")
        # self.layer_1 = RNNLayer(2, self,  name="layer_1")
        # self.layer_2 = RNNLayer(3, self,  name="layer_2")
        # self.layer_3 = RNNLayer(2, self,  name="layer_3")
        # self.layer_7 = RNNLayer(2, self,  name="layer_7")
        # self.layer_4 = RNNLayer(1, self,  name="layer_4")
        # self.layer_5 = RNNLayer(2, self,  name="layer_5")
        # self.layer_6 = RNNLayer(4, self,  name="layer_6")        
        # self.layer_8 = RNNLayer(3, self,  name="layer_8")
        # self.layer_9 = RNNLayer(2, self,  name="layer_9")
        # self.layer_10 = RNNLayer(1, self,  name="layer_10", is_output=True)
        
        # self.layer_1 = RNNLayer(2, self,  name="layer_1", act_func=np.array([[0,]]))
        
        self.layer_1 = RNNLayer(2, self,  name="layer_1", act_func="relu")
        # temp = np.array([[0,]])
        
        self.layer_2 = RNNLayer(3, self,  name="layer_2", act_func="relu")
        self.layer_3 = RNNLayer(2, self,  name="layer_3", act_func="relu")
        self.layer_7 = RNNLayer(2, self,  name="layer_7", act_func="relu")
        self.layer_4 = RNNLayer(1, self,  name="layer_4", act_func="relu")
        self.layer_5 = RNNLayer(2, self,  name="layer_5", act_func="relu")
        self.layer_6 = RNNLayer(4, self,  name="layer_6", act_func="relu")        
        self.layer_8 = RNNLayer(3, self,  name="layer_8", act_func="relu")
        self.layer_9 = RNNLayer(2, self,  name="layer_9", act_func="relu")
        self.layer_10 = RNNLayer(1, self,  name="layer_10", is_output=True)
        
    def define(self):
        # Figure P14.2 in the textbook
        self.layer_1([
            (self.input_layer, [0]),
        ]) 
        self.layer_2([
            (self.layer_1, [0]), 
            (self.layer_2, [1]),
        ])   
        self.layer_3([
            (self.layer_2, [0]), 
        ])
        self.layer_4([
            (self.layer_3, [1]), 
            (self.layer_4, [1]),
            (self.layer_7, [0]), 
        ])
        self.layer_5([
            (self.layer_4, [0]), 
        ])
        self.layer_6([
            (self.layer_2, [0]), 
            (self.layer_5, [0, 1]),
        ])
        self.layer_7([
            (self.layer_2, [0]), 
        ])
        self.layer_8([
            (self.layer_7, [0]), 
        ])
        self.layer_9([
            (self.layer_4, [0]),
            (self.layer_8, [1]), 
        ])
        self.layer_10([
            (self.layer_6, [0]), 
            (self.layer_9, [0]),
        ])    

class SimpleRNN4(RNNBuilder):
    def __init__(self, name):
        super(self.__class__, self).__init__(name)
        n_in = 2
        self.input_layer = InLayer(n_in, self, name="input_1", act_func="linear")
        self.layer_1 = RNNLayer(3, self,  name="layer_1", act_func="linear")
        self.layer_2 = RNNLayer(1, self,  name="layer_2", act_func="linear", is_output=True)
        
    def define(self):
        # The figure in Exercise E14.9 (ii) in the textbook
        self.layer_1([
            (self.input_layer, [0]),
            (self.layer_2, [1]), 
        ])
        self.layer_2([
            (self.layer_1, [0]), 
        ])

class SimpleRNN5(RNNBuilder):
    def __init__(self, name):
        super(self.__class__, self).__init__(name)
        n_in = 2
        self.input_layer = InLayer(n_in, self, name="input_1", act_func="sigmoid")
        self.layer_1 = RNNLayer(3, self,  name="layer_1", act_func="sigmoid")
        self.layer_2 = RNNLayer(1, self,  name="layer_2", act_func="sigmoid", is_output=True)
        
    def define(self):
        # The figure in Exercise E14.9 (ii) in the textbook, with sigmoid activation functions
        self.layer_1([
            (self.input_layer, [0]),
            (self.layer_2, [1]), 
        ])
        self.layer_2([
            (self.layer_1, [0]), 
        ])

class SimpleRNN6(RNNBuilder):
    def __init__(self, name):
        super(self.__class__, self).__init__(name)
        n_in = 2
        self.input_1 = InLayer(n_in, self, name="input_1")
        self.input_2 = InLayer(n_in, self, name="input_2")
        self.layer_1 = RNNLayer(3, self,  name="layer_1", act_func="relu")
        self.layer_3 = RNNLayer(5, self,  name="layer_3", act_func="sigmoid")
        self.layer_2 = RNNLayer(2, self,  name="layer_2", act_func="relu", is_output=True)
        
        
    def define(self):
        self.layer_1([
            (self.input_1, [0]),
            (self.layer_2, [1]),
        ])
        self.layer_2([
            (self.layer_1, [0]),
            (self.layer_3, [1, 3]),
        ])
        self.layer_3([
            (self.input_2, [0]),
            (self.layer_3, [1, 2]),
        ])
    
    def simple_forward_test(self):
        LW_1_2_d1 = np.array([
            [1, 4],
            [-0.5, 1],
            [3, 2],
        ])
        IW_1_1_d0 = np.array([
            [2, -5],
            [3, 7],
            [0.5, 1.25],
        ])
        b1 = np.array([0, 2, -1], dtype=np.float64)
        # a1 = np.array([0, 0, 0])
        
        LW_2_1_d0 = np.array([
            [1, -5, 4],
            [-0.5, 0.25, 2.5],
        ])
        LW_2_3_d1 = np.array([
            [4, -2, 1, 0.5, 3],
            [-0.75, 1, 2, 5, 1.5],
        ])
        LW_2_3_d3 = np.array([
            [1, 2, 6, -2, 0.2],
            [3, 5, -2, 1, 1.5],
        ])
        b2 = np.array([0.5, 0.25])
        # a2 = np.array([0, 0])
        
        LW_3_3_d1 = np.array([
            [1, 6, -4, 2.5, 0.75],
            [0.2, 0.6, -1.5, 2.25, 3],
            [0.75, -3, 4, 2.5, 1],
            [-5, -3, 2, 8, 9],
            [2, 1, 3, -5, 1.5],
        ])
        LW_3_3_d2 = np.array([
            [-1, -2, 2.5, 1.25, 0.5],
            [2.25, 3, 1.5, 1.75, 0.5],
            [-6, 1, 2, -0.25, 3],
            [13, -2, 3, 5, -1.5],
            [4, -6, 1.5, 1.25, 3],
        ])
        IW_3_2_d0 = np.array([
            [0.25, -1.5],
            [5, 3],
            [4, 7],
            [-1.75, 0.5],
            [2, 1],
        ])
        b3 = np.array([1.25, 0.75, 1.0, -0.25, -0.5])
        # a3 = np.array([0, 0, 0, 0, 0], )
        
        p1_all = np.array([
            [0.25, 1],
            [3, -1.25],
            [-2, 0.5],
            [1, 1],
        ])
        p2_all = np.array([
            [3, 1],
            [4.5, -3],
            [0.5, 0.25],
            [4, -2],
        ])
        
        print("")
        print("In method simple_forward_test")
        print("")
        
        print("First input")
        print("p1_all")
        print(p1_all)
        print("p2_all")
        print(p2_all)
        a1 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.float64)
        a2 = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ], dtype=np.float64)
        a3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float64)
        for step_idx in range(4):
            print("Step: %d" % (step_idx+1))
            p1 = p1_all[step_idx, :]
            p2 = p2_all[step_idx,:]
            if step_idx-1 >= 0:
                term1 = np.dot(LW_1_2_d1, a2[step_idx-1, :])
                term2 = np.dot(IW_1_1_d0, p1)
                a1[step_idx, :] = term1 + term2 + b1
            else:
                term1 = np.dot(LW_1_2_d1, np.zeros_like(a2[0, :]))
                # set_trace()
                term2 = np.dot(IW_1_1_d0, p1)
                a1[step_idx, :] = term1 + term2 + b1
            a1[step_idx, :] = eval_activation_func(a1[step_idx, :] , "relu")
            
            if step_idx-2 >= 0:
                term1 = np.dot(LW_3_3_d1, a3[step_idx-1, :])
                term2 = np.dot(LW_3_3_d2, a3[step_idx-2, :])
                term3 = np.dot(IW_3_2_d0, p2)
                a3[step_idx, :] = term1 + term2 + term3 + b3
            elif step_idx - 1 >= 0:
                term1 = np.dot(LW_3_3_d1, a3[step_idx-1, :])
                term2 = np.dot(LW_3_3_d2, np.zeros_like(a3[0, :]))
                term3 = np.dot(IW_3_2_d0, p2)
                a3[step_idx, :] = term1 + term2 + term3 + b3
            else:
                term1 = np.dot(LW_3_3_d1, np.zeros_like(a3[0, :]))
                term2 = np.dot(LW_3_3_d2, np.zeros_like(a3[0, :]))
                term3 = np.dot(IW_3_2_d0, p2)
                a3[step_idx, :] = term1 + term2 + term3 + b3
            # set_trace()
            a3[step_idx, :] = eval_activation_func(a3[step_idx, :] , "sigmoid")
            
            if step_idx - 3 >= 0:
                term1 = np.dot(LW_2_1_d0, a1[step_idx, :])
                term2 = np.dot(LW_2_3_d1, a3[step_idx-1, :])
                term3 = np.dot(LW_2_3_d3, a3[step_idx-3, :])
            elif step_idx - 1 >= 0:
                term1 = np.dot(LW_2_1_d0, a1[step_idx, :])
                term2 = np.dot(LW_2_3_d1, a3[step_idx-1, :])
                term3 = np.dot(LW_2_3_d3, np.zeros_like(a3[0, :]))
            else:
                term1 = np.dot(LW_2_1_d0, a1[step_idx, :])
                term2 = np.dot(LW_2_3_d1, np.zeros_like(a3[step_idx-1, :]))
                term3 = np.dot(LW_2_3_d3, np.zeros_like(a3[0, :]))
            a2[step_idx, :] = term1 + term2 + term3 + b2
            a2[step_idx, :] = eval_activation_func(a2[step_idx, :] , "relu")
            
            print("Output:")
            print(a2[step_idx, :])
    
        print("----------------------------")
            # set_trace()

@show_info 
def test_rnn_1():
    my_rnn = MyRecurrentNeuralNetwork("my_rnn")
    my_rnn.define()
    my_rnn.compile()
    my_rnn.summary()
    
    simple_rnn_1 = SimpleRNN1("simple_rnn_1")
    simple_rnn_1.define()
    simple_rnn_1.compile()
    simple_rnn_1.summary()
    
    simple_rnn_2 = SimpleRNN2("simple_rnn_2")
    simple_rnn_2.define()
    simple_rnn_2.compile()
    simple_rnn_2.summary()
    
    simple_rnn_3 = SimpleRNN3("simple_rnn_3")
    simple_rnn_3.define()
    simple_rnn_3.compile()
    simple_rnn_3.summary()

@show_info
def test_rnn_2():
    model = SimpleRNN2("simple_rnn_2")
    model.define()
    model.compile()
    
    model.iw[('layer_1', 'input_1')][0, 0, 0] = 2
    model.lw[('layer_1', 'layer_1')][0, 0, 0] = 0.5
    model.bias['layer_1'][0] = 0.0
    input_dict = {
        "input_1":  np.zeros((2, 3, 1))
    }
    input_dict["input_1"][0, 0, 0] = 2
    input_dict["input_1"][0, 1, 0] = 3
    input_dict["input_1"][0, 2, 0] = 2
    input_dict["input_1"][1, 0, 0] = 1
    input_dict["input_1"][1, 1, 0] = -2
    input_dict["input_1"][1, 2, 0] = -1
   
    out = model.forward(input_dict)
    print(out)
    # Correct output:
    # {'layer_1': array([[[ 4. ],
        # [ 8. ],
        # [ 8. ]],

       # [[ 2. ],
        # [-3. ],
        # [-3.5]]])}
    # set_trace()

@show_info
def test_rnn_3():
    model = SimpleRNN4("simple_rnn_4")
    model.define()
    model.compile()
    
    model.iw[('layer_1', 'input_1')][:,:,0] = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ])
    model.lw[('layer_1', 'layer_2')][:, :, 0] = np.array([
        [-1],
        [2],
        [-3],
    ])
    model.lw[('layer_2', 'layer_1')][:, :, 0] = np.array([
        [-2, 10, -3],
    ])
    for layer_name in model.bias:
        model.bias[layer_name] = np.zeros_like(model.bias[layer_name])
    input_dict = {
        "input_1":  np.zeros((1, 3, 2))
    }
    input_dict["input_1"][0, :, :] = np.array([
        [-0.5, 0.5],
        [1, 2],
        [2, -3],
    ])
    out = model.forward(input_dict)
    print(out)
    # Correct output
    # {'layer_2': array([[[2.5000e+00],
        # [1.2650e+02],
        # [3.8935e+03]]])}

    # set_trace()

@show_info
def test_rnn_4():
    model = SimpleRNN5("simple_rnn_5")
    model.define()
    model.compile()
    
    model.iw[('layer_1', 'input_1')][:,:,0] = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ])
    model.lw[('layer_1', 'layer_2')][:, :, 0] = np.array([
        [-1],
        [2],
        [-3],
    ])
    model.lw[('layer_2', 'layer_1')][:, :, 0] = np.array([
        [-2, 10, -3],
    ])
    for layer_name in model.bias:
        model.bias[layer_name] = np.zeros_like(model.bias[layer_name])
    input_dict = {
        "input_1":  np.zeros((1, 3, 2))
    }
    input_dict["input_1"][0, :, :] = np.array([
        [-0.5, 0.5],
        [1, 2],
        [2, -3],
    ])
    # set_trace()
    out = model.forward(input_dict)
    print(out)
    # Correct output
    # {'layer_2': array([[[0.95739713],
        # [0.99353249],
        # [0.54092538]]])}

    # set_trace()

@show_info
def test_rnn_5():
    # This function checks the correctness of the RNN's forward function 
    model = SimpleRNN6("simple_rnn_6")
    model.define()
    model.compile()
    model.simple_forward_test()
    
    print("Begin forward function")
    model.iw[("layer_1", "input_1")][: ,:, 0] = np.array([
        [2, -5],
        [3, 7],
        [0.5, 1.25],
    ])
    model.lw[("layer_1", "layer_2")][:, :, 0] = np.array([
        [1, 4],
        [-0.5, 1],
        [3, 2],
    ])
    
    model.iw[("layer_3", "input_2")][:, :, 0] = np.array([
        [0.25, -1.5],
        [5, 3],
        [4, 7],
        [-1.75, 0.5],
        [2, 1],
    ])
    model.lw[("layer_3", "layer_3")][:, :, 0] = np.array([
        [1, 6, -4, 2.5, 0.75],
        [0.2, 0.6, -1.5, 2.25, 3],
        [0.75, -3, 4, 2.5, 1],
        [-5, -3, 2, 8, 9],
        [2, 1, 3, -5, 1.5],
    ])
    model.lw[("layer_3", "layer_3")][:, :, 1] = np.array([
        [-1, -2, 2.5, 1.25, 0.5],
        [2.25, 3, 1.5, 1.75, 0.5],
        [-6, 1, 2, -0.25, 3],
        [13, -2, 3, 5, -1.5],
        [4, -6, 1.5, 1.25, 3],
    ])
    
    model.lw[("layer_2", "layer_1")][:, :, 0] = np.array([
        [1, -5, 4],
        [-0.5, 0.25, 2.5],
    ])
    model.lw[("layer_2", "layer_3")][:, :, 0] = np.array([
        [4, -2, 1, 0.5, 3],
        [-0.75, 1, 2, 5, 1.5],
    ])
    model.lw[("layer_2", "layer_3")][:, :, 1] = np.array([
        [1, 2, 6, -2, 0.2],
        [3, 5, -2, 1, 1.5],
    ])
    
    model.bias["layer_1"] = np.array([0, 2, -1], dtype=np.float64)
    model.bias["layer_2"] = np.array([0.5, 0.25])
    model.bias["layer_3"] = np.array([1.25, 0.75, 1.0, -0.25, -0.5])
    
    input_dict = {
        "input_1":  np.zeros((1, 4, 2)), 
        "input_2":  np.zeros((1, 4, 2)), 
    }
    input_dict["input_1"][0, :, :] = np.array([
        [0.25, 1],
        [3, -1.25],
        [-2, 0.5],
        [1, 1],
    ])
    input_dict["input_2"][0, :, :] = np.array([
        [3, 1],
        [4.5, -3],
        [0.5, 0.25],
        [4, -2],
    ])
    
    out = model.forward(input_dict)
    print("Result when running RNN.forward function")
    print(out)
    
    # Correct output:
    # {'layer_2': array([[[0.00000000e+00, 3.62500000e+00],
        # [2.71136802e+01, 7.87686797e+00],
        # [4.41122861e+02, 2.16520549e+02],
        # [8.30885069e+03, 3.75815755e+03]]])}

@show_info
def test_backprop_1():
    # This tests the BPTT results for the RNN shown in Figure 14.2 of the textbook
    model = SimpleRNN1("my_simple_rnn")
    model.define()
    model.compile()
    
    input_dict = {
        # "input_1":  np.zeros((1, 4, 1)), 
        "input_1":  np.zeros((1, 3, 1)), 
    }
    
    input_dict["input_1"][0, :, :] = np.array([
        [1],
        [2],
        [3],
        # [5],
    ])
    gt_dict = {
        "layer_1": np.zeros((1, 3, 1)),
    }
    gt_dict["layer_1"][0, :, :] = np.array([
        [1.5],
        [-3],
        [1],
    ])
    model.bias['layer_1'][0] = 0
    model.iw[('layer_1', 'input_1')] = np.array([[[ 0.15397634, -0.78608313,  1.0679081 ]]])
    # set_trace()
    pred_dict = model.forward(input_dict)
    loss_dict, loss_grad_dict = eval_loss_func_rnn(gt_dict, pred_dict)
    model.backward(loss_grad_dict, input_dict)
    print("Gradient calculated by model:")
    print(model.iw_grad)
    print("Correct gradient:")
    err = gt_dict["layer_1"][0, :, :] - pred_dict["layer_1"][0, :, :]
    err = err[:, 0]
    p1 = input_dict["input_1"][0, :, :]
    p1 = p1[:, 0]
    correct_grad = np.array([
        -2*(err[0]*p1[0] + err[1]*p1[1] + err[2]*p1[2]),
        -2*(err[0]*0       + err[1]*p1[0] + err[2]*p1[1]),
        -2*(err[0]*0       + err[1]*0       + err[2]*p1[0]),
    ]) # You can derive this by hand
    print(correct_grad)
    # Correct output:
    # Gradient calculated by model:
    # {('layer_1', 'input_1'): array([[[ 1.14145604,  0.87442254, -2.08465828]]])}
    # Correct gradient:
    # [ 1.14145604  0.87442254 -2.08465828]

@show_info
def test_backprop_2():
    # This tests the implementation based on the calculations shown in Problem P14.1 in the textbook 
    model = SimpleRNN3("my_simple_rnn")
    model.define()
    model.compile()
    
    input_dict = {
        "input_1":  np.random.rand(2, 4, 3)
    }
    gt_dict = {
        "layer_10": np.zeros((1, 4, 1)),        
    }
    gt_dict["layer_10"][0, :, :] = np.array([
        [1.5],
        [-3],
        [1],
        [6],
    ])
    pred_dict = model.forward(input_dict)
    # set_trace()
    loss_dict, loss_grad_dict = eval_loss_func_rnn(gt_dict, pred_dict)
    model.backward(loss_grad_dict, input_dict)

@show_info
def test_backprop_3():
    # This tests the BPTT results for the RNN shown in Figure 14.4 of the textbook
    model = SimpleRNN2("my_simple_rnn")
    model.define()
    model.compile()
    
    input_dict = {
        # "input_1":  np.zeros((1, 4, 1)), 
        "input_1":  np.zeros((1, 3, 1)), 
    }
    
    input_dict["input_1"][0, :, :] = np.array([
        [1],
        [2],
        [3],
        # [5],
    ])
    gt_dict = {
        # "layer_1": np.zeros((1, 4, 1)),        
        "layer_1": np.zeros((1, 3, 1)),
    }
    gt_dict["layer_1"][0, :, :] = np.array([
        [1.5],
        [-3],
        [1],
        # [6],
    ])
    model.lw[('layer_1', 'layer_1')] = np.array([[[0.777298]]])
    model.iw[('layer_1', 'input_1')] = np.array([[[-0.16155633]]])
    model.bias['layer_1'] = np.array([0.0])
    pred_dict = model.forward(input_dict)
    loss_dict, loss_grad_dict = eval_loss_func_rnn(gt_dict, pred_dict)
    model.backward(loss_grad_dict, input_dict)
    
    print("Gradient calculated by model:")
    print("divF[LW^{1,1}(1)]:")
    print(model.lw_grad)
    print("difvF[IW^{1,1}(0)]:")
    print(model.iw_grad)
    
    print("Correct gradient:")
    err = gt_dict["layer_1"][0, :, :] - pred_dict["layer_1"][0, :, :]
    err = err[:, 0]
    p1 = input_dict["input_1"][0, :, :]
    p1 = p1[:, 0]
    lw11 = model.lw[('layer_1', 'layer_1')][0,0,0]
    a0 = 0
    a1 = pred_dict['layer_1'][0,0,0]
    a2 = pred_dict['layer_1'][0,1,0]
    # You can derive this by hand
    correct_grad_lw_timestep_1 = a0*((-2)*err[0] + lw11*(-2*err[1]) + (lw11*lw11)*(-2*err[2])) + a1*(-2*err[1] + lw11*(-2*err[2])) + a2*(-2)*err[2]
    correct_grad_iw_timestep_0 = p1[0]*(-2*err[0]+lw11*-2*err[1] + (lw11*lw11)*(-2)*err[2]) + p1[1]*(-2*err[1] + lw11*(-2)*err[2]) + p1[2]*(-2)*err[2]
    print("divF[LW^{1,1}(1)]:")
    print(correct_grad_lw_timestep_1)
    print("difvF[IW^{1,1}(0)]:")
    print(correct_grad_iw_timestep_0)
    
    # Correct output:
    # Gradient calculated by model:
    # divF[LW^{1,1}(1)]:
    # {('layer_1', 'layer_1'): array([[[1.28140354]]])}
    # difvF[IW^{1,1}(0)]:
    # {('layer_1', 'input_1'): array([[[-8.06822118]]])}
    # Correct gradient:
    # divF[LW^{1,1}(1)]:
    # 1.2814035420041168
    # difvF[IW^{1,1}(0)]:
    # -8.068221177900085

# def 
@show_info
def test_backprop_4():
    # This function tests the correctness of the implementation of the BPTT algorithm
    # by using the centre difference formula:
    # dF( x ) / dx = [f( x + h ) - f( x - h )] / (2*h)
    # which is a second order accurate method with error bounded by O(h^2)
    model = SimpleRNN6("my_simple_rnn")
    model.define()
    model.compile()
    
    eps = 1e-5
    # eps = 1e-4
    # eps = 1e-6
    
    batch_size = 1
    n_step = 7
    n_features = 2
    input_dict = {
        "input_1":  np.random.rand(batch_size, n_step, n_features), 
        "input_2":  np.random.rand(batch_size, n_step, n_features), 
    }
    gt_dict =  {
        "layer_2": np.random.rand(batch_size, n_step, 2),
    }
    dF_numerical_list = []
    
    pred_dict = model.forward(input_dict)
    loss_dict, loss_grad_dict = eval_loss_func_rnn(gt_dict, pred_dict)
    model.backward(loss_grad_dict, input_dict)

    # This part can be made much more efficiently but this is just a test function
    # so there is little need for that
    model_upper = deepcopy(model)
    model_lower = deepcopy(model)
    
    for (layer_m, layer_l) in model.lw:
        (layer_m_dim, layer_l_dim, n_delay) = model.lw[(layer_m, layer_l)].shape
        df_numerical = np.zeros((layer_m_dim, layer_l_dim, n_delay))
        for m_idx in range(layer_m_dim):
            for l_idx in range(layer_l_dim):
                for delay_idx in range(n_delay):
                    x_orig = model.lw[(layer_m, layer_l)][m_idx, l_idx, delay_idx]
                    model_upper.lw[(layer_m, layer_l)][m_idx, l_idx, delay_idx] = x_orig + eps
                    model_lower.lw[(layer_m, layer_l)][m_idx, l_idx, delay_idx] = x_orig - eps
                    
                    pred_upper_dict = model_upper.forward(input_dict)
                    loss_upper_dict, _ = eval_loss_func_rnn(gt_dict, pred_upper_dict)
                    f_plus = loss_upper_dict['layer_2']
                    
                    pred_lower_dict = model_lower.forward(input_dict)
                    loss_lower_dict, _ = eval_loss_func_rnn(gt_dict, pred_lower_dict)
                    f_minus = loss_lower_dict['layer_2']
                    
                    df_numerical[m_idx, l_idx, delay_idx] = (f_plus-f_minus)/(2*eps)
                    
                    model_upper.lw[(layer_m, layer_l)][m_idx, l_idx, delay_idx] = x_orig
                    model_lower.lw[(layer_m, layer_l)][m_idx, l_idx, delay_idx] = x_orig

        df_real = model.lw_grad[(layer_m, layer_l)]
        relative_err = np.sum((df_numerical-df_real) / (np.abs(df_numerical)+eps))
        print(("Relative error between true gradient and gradient found via "
            "centre difference formula of layer-to-layer weight from %s to %s:") % (layer_l, layer_m,))
        print(relative_err)

    for (layer_m, layer_l) in model.iw:
        (layer_m_dim, layer_l_dim, n_delay) = model.iw[(layer_m, layer_l)].shape
        df_numerical = np.zeros((layer_m_dim, layer_l_dim, n_delay))
        for m_idx in range(layer_m_dim):
            for l_idx in range(layer_l_dim):
                for delay_idx in range(n_delay):
                    x_orig = model.iw[(layer_m, layer_l)][m_idx, l_idx, delay_idx]
                    model_upper.iw[(layer_m, layer_l)][m_idx, l_idx, delay_idx] = x_orig + eps
                    model_lower.iw[(layer_m, layer_l)][m_idx, l_idx, delay_idx] = x_orig - eps
                    
                    pred_upper_dict = model_upper.forward(input_dict)
                    loss_upper_dict, _ = eval_loss_func_rnn(gt_dict, pred_upper_dict)
                    f_plus = loss_upper_dict['layer_2']
                    
                    pred_lower_dict = model_lower.forward(input_dict)
                    loss_lower_dict, _ = eval_loss_func_rnn(gt_dict, pred_lower_dict)
                    f_minus = loss_lower_dict['layer_2']
                    
                    df_numerical[m_idx, l_idx, delay_idx] = (f_plus-f_minus)/(2*eps)
                    
                    model_upper.iw[(layer_m, layer_l)][m_idx, l_idx, delay_idx] = x_orig
                    model_lower.iw[(layer_m, layer_l)][m_idx, l_idx, delay_idx] = x_orig

        df_real = model.iw_grad[(layer_m, layer_l)]
        relative_err = np.sum((df_numerical-df_real) / (np.abs(df_numerical)+eps))
        print(("Relative error between true gradient and gradient found via "
            "centre difference formula of input-to-layer weight from %s to %s:") % (layer_l, layer_m,))
        print(relative_err)

    for layer_m in model.bias:
        (layer_m_dim, ) = model.bias[layer_m].shape
        df_numerical = np.zeros((layer_m_dim))
        for m_idx in range(layer_m_dim):
            x_orig = model.bias[layer_m][m_idx]            
            model_upper.bias[layer_m][m_idx] = x_orig + eps
            model_lower.bias[layer_m][m_idx]  = x_orig - eps
            
            pred_upper_dict = model_upper.forward(input_dict)
            loss_upper_dict, _ = eval_loss_func_rnn(gt_dict, pred_upper_dict)
            f_plus = loss_upper_dict['layer_2']
            
            pred_lower_dict = model_lower.forward(input_dict)
            loss_lower_dict, _ = eval_loss_func_rnn(gt_dict, pred_lower_dict)
            f_minus = loss_lower_dict['layer_2']
            
            df_numerical[m_idx] = (f_plus-f_minus)/(2*eps)
            
            model_upper.bias[layer_m][m_idx]  = x_orig
            model_lower.bias[layer_m][m_idx]  = x_orig
            # set_trace()
        df_real = model.bias_grad[layer_m]
        relative_err = np.sum((df_numerical-df_real) / (np.abs(df_numerical)+eps))
        print(("Relative error between true gradient and gradient found via "
            "centre difference formula of the bias for %s:") % (layer_m,))
        print(relative_err)
    
    # Example of correct outputs:
    # Relative error between true gradient and gradient found via centre difference formula of layer-to-layer weight from layer_2 to layer_1:
    # 9.215803797788378e-11
    # Relative error between true gradient and gradient found via centre difference formula of layer-to-layer weight from layer_1 to layer_2:
    # 8.815791960055927e-10
    # Relative error between true gradient and gradient found via centre difference formula of layer-to-layer weight from layer_3 to layer_2:
    # 1.0072548107549477e-08
    # Relative error between true gradient and gradient found via centre difference formula of layer-to-layer weight from layer_3 to layer_3:
    # 2.554254878418801e-09
    # Relative error between true gradient and gradient found via centre difference formula of input-to-layer weight from input_1 to layer_1:
    # 3.809334958591521e-11
    # Relative error between true gradient and gradient found via centre difference formula of input-to-layer weight from input_2 to layer_3:
    # -1.3749834423413785e-09
    # Relative error between true gradient and gradient found via centre difference formula of the bias for layer_1:
    # -6.264313045058738e-11
    # Relative error between true gradient and gradient found via centre difference formula of the bias for layer_3:
    # 3.837315435328144e-10
    # Relative error between true gradient and gradient found via centre difference formula of the bias for layer_2:
    # -1.2160742435181199e-11
    # set_trace()
    
def main():
    # (train_data, train_labels), (test_data, test_labels) = tf.compat.v1.keras.datasets.imdb.load_data(num_words=10000)
    test_rnn_1()
    test_rnn_2()
    test_rnn_3()
    test_rnn_4()
    test_rnn_5()

    test_backprop_1()
    test_backprop_2()
    test_backprop_3()
    test_backprop_4()
     
if __name__ == '__main__':
    main()
