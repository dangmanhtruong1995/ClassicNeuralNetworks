import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
import random
import copy

# from neural_network_regression import RegressionNeuralNetwork
from utils import eval_activation_func, eval_activation_func_gradient, eval_loss_func, unflatten_from_vector

import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)
random.seed(72)

class NeuralNetwork:
    def __init__(self):
        pass

class Perceptron(NeuralNetwork):
    def __init__(self, in_dim, out_dim, activation_func="linear", learning_rate=0.03):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        
        # self.weight = np.random.normal(size=(out_dim, in_dim))
        # self.bias = np.random.normal(size=(out_dim))
        
        self.weight = np.zeros((out_dim, in_dim))
        self.bias = np.zeros((out_dim))
      
        self.weight_grad = np.zeros((out_dim, in_dim))
        self.bias_grad = np.zeros((out_dim))
        
    def eval(self, x_in):
        out_mat_1 = np.dot(self.weight, x_in)
        out_mat = out_mat_1.squeeze() + self.bias.squeeze()
        out = eval_activation_func(out_mat, self.activation_func)
        return out
                
    def train(self, X_train, y_train, n_epoch):
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
                
                if np.size(y_inst) == 1:
                    y_inst = y_inst[:, np.newaxis]                
               
                # Forward pass
                out = self.eval(x_inst)
                err = y_inst - out
                err = np.squeeze(err)
                
                # Update
                self.weight += 2*self.learning_rate*err*x_inst
                self.bias += 2*self.learning_rate*err
                
                # Calculate sum of square loss
                loss += np.sum(err**2) # MSE   
                
            loss /= (1.0*n_inst)
            loss_list.append(loss)        

        return loss_list

def get_experiment_data():
    data_range = np.arange(-1, 1, 0.1)
    n_point = np.size(data_range)
    
    sin_data = np.sin(data_range*np.pi/5.0).tolist()
    sin_data_delayed = copy.deepcopy(sin_data)
    sin_data_delayed.insert(0, 0)
    sin_data_delayed = sin_data_delayed[:-1]
    
    X_train = np.zeros((n_point, 2))
    X_train[:, 0] = np.array(sin_data)
    X_train[:, 1] = np.array(sin_data_delayed)
    
    y_train = np.array(sin_data)    
    y_train = y_train[:, np.newaxis]
    
    return (X_train, y_train)

def run_experiment():
    (X_train, y_train) = get_experiment_data()
    (n_inst, n_features) = X_train.shape

    net = Perceptron(n_features, 1, "linear", learning_rate=0.5) # From 0.82 the error diverges
    n_epoch = 40
    loss_list = net.train(X_train, y_train, n_epoch)

    plt.plot(range(1, len(loss_list)+1), loss_list, color="blue", label="Sum of square error")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Time step")
    plt.title("Adaptive predictor experiment")
    plt.show()

def main():    
    run_experiment()


if __name__ == "__main__":
    main()
