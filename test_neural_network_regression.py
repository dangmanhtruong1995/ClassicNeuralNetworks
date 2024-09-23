import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
# import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from neural_network_regression import eval_activation_func, RegressionNeuralNetwork, eval_loss_func


def test_layer_sizes():
    print("Function test_layer_sizes")
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    
    print('1st net: RegressionNeuralNetwork(3, [2, 4, 5], 6, ["sigmoid", "sigmoid", "sigmoid", "linear"], 0.01)')
    net = RegressionNeuralNetwork(3, [2, 4, 5], 6, ["sigmoid", "sigmoid", "sigmoid", "linear"], optimizer_params)
    net.print_layer_sizes()

    print('2nd net: RegressionNeuralNetwork(3, [10, 4, 5], 8, ["sigmoid", "sigmoid", "sigmoid", "linear"], 0.01)')
    net = RegressionNeuralNetwork(3, [10, 4, 5], 8, ["sigmoid", "sigmoid", "sigmoid", "linear"], optimizer_params)
    net.print_layer_sizes()
    
    print('3rd net: RegressionNeuralNetwork(1, [4, 4], 1, ["sigmoid", "sigmoid", "linear"], 0.01)')
    net = RegressionNeuralNetwork(1, [4, 4], 1, ["sigmoid", "sigmoid", "linear"], optimizer_params)    
    net.print_layer_sizes()

    print("")

def test_eval_1():
    print("Function test_eval_1")
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    
    print('Net: RegressionNeuralNetwork(1, [2], 1, ["sigmoid", "linear"], 0.01)')
    net = RegressionNeuralNetwork(1, [2], 1, ["sigmoid", "linear"], optimizer_params)
    net.layer_list[0]["weight"] = np.array(
        [[-0.27],
         [-0.41],
        ])
    net.layer_list[0]["bias"] = np.array(
        [[-0.48],
         [-0.13],
        ])
    net.layer_list[1]["weight"] = np.array(
        [[0.09, -0.17],
        ])
    net.layer_list[1]["bias"] = np.array(
        [[0.48],
        ])
    x_in = np.array([1])
    net.eval(x_in)
    out_list = net.out_list
    for layer_idx, out in enumerate(out_list):
        print("Layer %d:" % (layer_idx+1))
        print(out)

    """ Correct output:
    Layer 1:
    [[0.3208213 ]
     [0.36818758]]
    Layer 2:
    [[0.44628203]]
    """

    print("")

def test_eval_2():
    print("Function test_eval_2")
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    
    print('Net: RegressionNeuralNetwork(1, [1], 1, ["sigmoid", "linear"], 0.01)')
    net = RegressionNeuralNetwork(1, [1], 1, ["sigmoid", "linear"], optimizer_params)
    net.layer_list[0]["weight"] = np.array(
        [[1],
        ])
    net.layer_list[0]["bias"] = np.array(
        [[1],
        ])
    net.layer_list[1]["weight"] = np.array(
        [[-2],
        ])
    net.layer_list[1]["bias"] = np.array(
        [[1],
        ])
    x_in = np.array([1])
    net.eval(x_in)
    out_list = net.out_list
    for layer_idx, out in enumerate(out_list):
        print("Layer %d:" % (layer_idx+1))
        print(out)

    """ Correct output:
    Layer 1:
    [[0.88079708]]
    Layer 2:
    [[-0.76159416]]
    """

    print("")


def test_eval_3():
    print("Function test_eval_3")
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    
    print('Net: RegressionNeuralNetwork(1, [1], 1, ["tanh", "tanh"], 0.01)')
    net = RegressionNeuralNetwork(1, [1], 1, ["tanh", "tanh"], optimizer_params)
    net.layer_list[0]["weight"] = np.array(
        [[-1],
        ])
    net.layer_list[0]["bias"] = np.array(
        [[1],
        ])
    net.layer_list[1]["weight"] = np.array(
        [[-2],
        ])
    net.layer_list[1]["bias"] = np.array(
        [[1],
        ])
    x_in = np.array([-1])
    net.eval(x_in)
    out_list = net.out_list
    for layer_idx, out in enumerate(out_list):
        print("Layer %d:" % (layer_idx+1))
        print(out)

    """ Correct output:
    Layer 1:
    [[0.96402758]]
    Layer 2:
    [[-0.72968586]]
    """

    print("")


def test_loss_func_1():
    print("Function test_loss_func_1")
    
    y_true = np.array(
        [
         [1],
         [0],
        ]
    )
    y_pred = np.array(
        [
         [0.8],
         [0.6],
        ]
    )    
    loss = eval_loss_func(y_true, y_pred, "mse")
    print(y_true)
    print(y_pred)
    print(loss)
    
    print("")


def test_loss_func_2():
    print("Function test_loss_func_2")
    
    y_true = np.array(
        [
         [1, 0, 0],
         [0, 0, 1],
        ]
    )
    y_pred = np.array(
        [
         [0.8, 0.1, 0.1],
         [0.6, 0.1, 0.3],
        ]
    )    
    loss = eval_loss_func(y_true, y_pred, "mse")
    print(y_true)
    print(y_pred)
    print(loss)
    
    print("")


def test_backprop_1():
    print("Function test_backprop_1")
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    
    print('Net: RegressionNeuralNetwork(1, [2], 1, ["sigmoid", "linear"], 0.01)')
    net = RegressionNeuralNetwork(1, [2], 1, ["sigmoid", "linear"], optimizer_params)
    net.layer_list[0]["weight"] = np.array(
        [[-0.27],
         [-0.41],
        ])
    net.layer_list[0]["bias"] = np.array(
        [[-0.48],
         [-0.13],
        ])
    net.layer_list[1]["weight"] = np.array(
        [[0.09, -0.17],
        ])
    net.layer_list[1]["bias"] = np.array(
        [[0.48],
        ])
    x_in = np.array(
        [1],               
    )
    net.eval(x_in)
    out_list = net.out_list
    
    y_true = np.array(
        [1.7071067811865475],    
    )
    
    net.backprop(y_true)
    for layer_idx in reversed(range(0, net.n_layer)):
        print("Sensitivity  of layer %d:" % (layer_idx+1))
        print(net.layer_list[layer_idx]["sensitivity"])
 
    """ Correct output:
    Sensitivity  of layer 2:
    [-2.52164951]
    Sensitivity  of layer 1:
    [[-0.04945093]
     [ 0.09972199]]  
    """   
 
    print("")


def test_backprop_2():
    print("Function test_backprop_2")
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    
    print('Net: RegressionNeuralNetwork(1, [1], 1, ["sigmoid", "linear"], 0.01)')
    net = RegressionNeuralNetwork(1, [1], 1, ["sigmoid", "linear"], optimizer_params)
    net = RegressionNeuralNetwork(1, [1], 1, ["sigmoid", "linear"], optimizer_params)
    net.layer_list[0]["weight"] = np.array(
        [[1],
        ])
    net.layer_list[0]["bias"] = np.array(
        [[1],
        ])
    net.layer_list[1]["weight"] = np.array(
        [[-2],
        ])
    net.layer_list[1]["bias"] = np.array(
        [[1],
        ])
    x_in = np.array(        
        [1],        
    )
    net.eval(x_in)
    out_list = net.out_list
    
    y_true = np.array(
        [1.0],
    )    
    net.backprop(y_true)
    for layer_idx in reversed(range(0, net.n_layer)):
        print("Sensitivity  of layer %d:" % (layer_idx+1))
        print(net.layer_list[layer_idx]["sensitivity"])
 
    """ Correct output:
    Sensitivity  of layer 2:
    [-3.52318831]
    Sensitivity  of layer 1:
    [0.73982435]
    """   
 
    print("")


def test_update_gradient_1():
    print("Function test_update_gradient_1")
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    
    print('Net: RegressionNeuralNetwork(1, [2], 1, ["sigmoid", "linear"], 0.01)')
    net = RegressionNeuralNetwork(1, [2], 1, ["sigmoid", "linear"], optimizer_params)
    net.layer_list[0]["weight"] = np.array(
        [[-0.27],
         [-0.41],
        ])
    net.layer_list[0]["bias"] = np.array(
        [[-0.48],
         [-0.13],
        ])
    net.layer_list[1]["weight"] = np.array(
        [[0.09, -0.17],
        ])
    net.layer_list[1]["bias"] = np.array(
        [[0.48],
        ])
    x_in = np.array(
        [1],               
    )
    net.eval(x_in)
    out_list = net.out_list
    
    y_true = np.array(
        [1.7071067811865475],    
    )
    
    net.backprop(y_true)
    net.update_gradient(y_true)
    for layer_idx in reversed(range(0, net.n_layer)):
        print("Weight gradient of layer %d:" % (layer_idx+1))
        print(net.layer_list[layer_idx]["weight_grad"])
        print("Bias gradient of layer %d:" % (layer_idx+1))
        print(net.layer_list[layer_idx]["bias_grad"])
 
    """ Correct output:
    Weight gradient of layer 2:
    [[-0.80899887 -0.92844004]]
    Bias gradient of layer 2:
    [-2.52164951]
    Weight gradient of layer 1:
    [[-0.04945093]
     [ 0.09972199]]
    Bias gradient of layer 1:
    [-0.04945093  0.09972199]
    """   
 
    print("")
    
    
def test_update_gradient_2():
    print("Function test_update_gradient_2")
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    
    print('Net: RegressionNeuralNetwork(1, [1], 1, ["sigmoid", "linear"], 0.01)')
    net = RegressionNeuralNetwork(1, [1], 1, ["sigmoid", "linear"], optimizer_params)
    net.layer_list[0]["weight"] = np.array(
        [[1],
        ])
    net.layer_list[0]["bias"] = np.array(
        [[1],
        ])
    net.layer_list[1]["weight"] = np.array(
        [[-2],
        ])
    net.layer_list[1]["bias"] = np.array(
        [[1],
        ])
    x_in = np.array(        
        [1],        
    )
    net.eval(x_in)
    out_list = net.out_list
    
    y_true = np.array(
        [1.0],
    )    
 
    net.backprop(y_true)
    net.update_gradient(y_true)
    for layer_idx in reversed(range(0, net.n_layer)):
        print("Weight gradient of layer %d:" % (layer_idx+1))
        print(net.layer_list[layer_idx]["weight_grad"])
        print("Bias gradient of layer %d:" % (layer_idx+1))
        print(net.layer_list[layer_idx]["bias_grad"])
 
    """ Correct output:
    Weight gradient of layer 2:
    [[-3.10321397]]
    Bias gradient of layer 2:
    [-3.52318831]
    Weight gradient of layer 1:
    [[0.73982435]]
    Bias gradient of layer 1:
    [0.73982435]
    """   
 
    print("")
 

def run_tests():
    test_layer_sizes()
    
    test_eval_1()
    test_eval_2()
    test_eval_3()    
    
    test_loss_func_1()
    test_loss_func_2()
    
    test_backprop_1()
    test_backprop_2()
    
    test_update_gradient_1()
    test_update_gradient_2()


def main():
    run_tests()


if __name__ == "__main__":
    main()
