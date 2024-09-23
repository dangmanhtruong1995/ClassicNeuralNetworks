import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt

from neural_network_regression import RegressionNeuralNetwork
from utils import get_total_number_of_parameters, flatten_into_vector, unflatten_from_vector


def test_flatten_into_vector():
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    net = RegressionNeuralNetwork(2, [5, 3], 1, ["sigmoid", "sigmoid", "linear"], 
        optimizer_params, optimizer_name="gradient_descent")
    net.layer_list[0]["weight_grad"] = np.array(
        [[ 1,  2],
         [ 3,  4],
         [ 5,  6],
         [ 7,  8],
         [ 9, 10],
        ]
    )
    net.layer_list[0]["bias_grad"] = np.array(
        [-1,
         -2,
         -3,
         -4,
         -5,
        ]
    )

    net.layer_list[1]["weight_grad"] = np.array(
        [[11,  12, 13, 14, 15],
         [16,  17, 18, 19, 20],
         [21,  22, 23, 24, 25]
        ]
    )
    net.layer_list[1]["bias_grad"] = np.array(
        [-6,
         -7,
         -8,         
        ]
    )
    
    net.layer_list[2]["weight_grad"] = np.array(
        [[26,  27, 28],         
        ]
    )
    net.layer_list[2]["bias_grad"] = np.array(
        [-9],        
    )
    
    net.print_layer_sizes()
    net.print_layers()
    net.print_layer_grad_sizes()    
    net.print_grad_layers()
    
    n_dim = get_total_number_of_parameters(net)
    print("Total number of parameters = %d" % (n_dim))
    
    # arr = flatten_into_vector(net, is_grad=True)
    arr = flatten_into_vector(net, is_grad=False)
    print(arr)

def test_unflatten_from_vector():
    arr = np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., -1., -2., -3., -4., -5., 11., 12., 13.,
        14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., -6., -7., -8., 26., 27., 28.,
     -9.])
    optimizer_params = {
        "learning_rate": 0.01, 
    }
    net = RegressionNeuralNetwork(2, [5, 3], 1, ["sigmoid", "sigmoid", "linear"], 
        optimizer_params, optimizer_name="gradient_descent")
    
    unflatten_from_vector(net, arr, is_grad=False)
    # unflatten_from_vector(net, arr, is_grad=True)
    
    net.print_layer_sizes()
    net.print_layers()
    net.print_layer_grad_sizes()    
    net.print_grad_layers()
    
    
def main():
    test_flatten_into_vector()
    
    print("")
    print("")
    print("")
    
    test_unflatten_from_vector()
    
if __name__ == "__main__":
    main()    
