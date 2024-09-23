import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt

from neural_network_regression import RegressionNeuralNetwork
from utils import log_time
from config import get_config

import logging
logging.basicConfig(level=get_config()["debug_mode"])
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def create_data(): 
    X_train = np.arange(-2, 3, 0.001)
    X_train = X_train[:, np.newaxis]
    y_train = np.zeros(X_train.shape[0])
    for inst_idx in range(X_train.shape[0]):
        inst = X_train[inst_idx, :]
        y_train[inst_idx] = 1+np.sin(1.5*np.pi*inst)
    y_train = y_train[:, np.newaxis]    
    return (X_train, y_train)

@log_time
def test_gradient_descent():
    print("Test with gradient descent")
    (X_train, y_train) = create_data()
    optimizer_params = {
        "learning_rate": 0.1, 
    }
    net = RegressionNeuralNetwork(1, [10], 1, ["sigmoid", "linear"], optimizer_params, optimizer_name="gradient_descent")
    n_epoch = 100000
    # n_epoch = 10
    # set_trace()
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red")
    plt.plot(base_list, y_pred, color="blue")
    plt.title("Neural network training using gradient descent.")
    plt.show()

@log_time
def test_conjugate_gradient():
    print("Test with conjugate gradient descent using the Polak-Ribiere method")
    (X_train, y_train) = create_data()
    optimizer_params = {        
        # "method": "fletcher_and_reeves",
        # "method": "hestenes_and_stiefel",
        "method": "polak_and_ribiere",
    }
    net = RegressionNeuralNetwork(1, [5, 2], 1, ["sigmoid", "linear"],
        optimizer_params, optimizer_name="conjugate_gradient")
    # n_epoch = 4000
    n_epoch = 2000
    # n_epoch = 2
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out, _ = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red")
    plt.plot(base_list, y_pred, color="blue")
    plt.title("Neural network training using conjugate gradient descent with the Polak-Ribiere method.")
    plt.show()

@log_time
def test_conjugate_gradient_2():
    print("Test with conjugate gradient descent using the Hestenes-Stiefel method")
    (X_train, y_train) = create_data()
    optimizer_params = {
        # "learning_rate": 0.1, 
        # "method": "fletcher_and_reeves",
        "method": "hestenes_and_stiefel",
        # "method": "polak_and_ribiere",
    }
    net = RegressionNeuralNetwork(1, [5, 2], 1, ["sigmoid", "sigmoid", "linear"],
        optimizer_params, optimizer_name="conjugate_gradient")
    # n_epoch = 10000
    n_epoch = 5000
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out, _ = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red")
    plt.plot(base_list, y_pred, color="blue")
    plt.title("Neural network training using conjugate gradient descent with the Hestenes-Stiefel method.")
    plt.show()


@log_time
def test_conjugate_gradient_3():
    print("Test with conjugate gradient descent using the Fletcher-Reeves method")
    (X_train, y_train) = create_data()
    optimizer_params = {
        # "learning_rate": 0.1, 
        "method": "fletcher_and_reeves",
        # "method": "hestenes_and_stiefel",
        # "method": "polak_and_ribiere",
    }
    net = RegressionNeuralNetwork(1, [5, 2], 1, ["sigmoid", "sigmoid", "linear"],
        optimizer_params, optimizer_name="conjugate_gradient")
    # n_epoch = 10000
    n_epoch = 5000
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out, _ = net.eval(X_train[inst_idx, :])
        out = out[0][0]
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red")
    plt.plot(base_list, y_pred, color="blue")
    plt.title("Neural network training using conjugate gradient descent with the Hestenes-Stiefel method.")
    plt.show()


@log_time
def test_adam():
    print("Test with the Adam algorithm")
    (X_train, y_train) = create_data()
    optimizer_params = {
        "learning_rate": 0.001,         
        "beta1": 0.9,
        "beta2": 0.999,        
    }
    # net = RegressionNeuralNetwork(1, [10], 1, ["sigmoid", "linear"], optimizer_params, optimizer_name="Adam")
    # net = RegressionNeuralNetwork(1, [50, 50], 1, ["LeakyRelu", "LeakyRelu", "linear"], optimizer_params, optimizer_name="Adam")

    n_epoch = 5000
    # n_epoch = 3
    net.train(X_train, y_train, n_epoch)
    
    y_pred = []
    for inst_idx in range(X_train.shape[0]):
        out, _ = net.eval(X_train[inst_idx, :])
        # set_trace()
        out = out[0][0]
        # set_trace()
        y_pred.append(out)
    
    y_train = y_train.tolist()
    y_train = [elem[0] for elem in y_train]
    base_list = list(range(X_train.shape[0]))
    
    plt.plot(base_list, y_train, color="red")
    plt.plot(base_list, y_pred, color="blue")
    plt.title("Neural network training using the Adam algorithm.")
    plt.show()

    
def main():   
    # Vanilla gradient descent needs 100000 epochs to converge on this small problem (!)
    # while conjugate gradient requires just around: 
    # - 3000 epochs (fletcher_and_reeves), if you run at 1000 epochs then sometimes it will not fit properly, 
    #   and sometimes the loss will go to infinity, probably due to the procedure of division in calculating beta
    # - Other methods: Same. 
    # - I tried gradient clipping, clipping the beta value, etc. but it did not help.
    #   Sometimes the values explode but then the algorithm gracefully returns, but other times it fails to do so. 
    # - If I use 2-hidden-layer network, then the polak_and_ribiere method usually becomes unstable and diverges
    # Adam: Converges at around 5000 epochs
    
    # test_gradient_descent()
    # test_conjugate_gradient()
    # test_conjugate_gradient_2()  
    test_conjugate_gradient_3()  
    # test_adam()


if __name__ == "__main__":
    main()    

