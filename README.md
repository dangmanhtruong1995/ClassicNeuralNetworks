# Neural networks

Implementations of a number of classic neural network and training algoritms.

- **adaptive_predictor.py: Impelemnt a simple Perceptron network. The network tries to predict based on results of two previous time steps.

- **bayesian_regularized_neural_network.py: Implement a Bayesian regularized neural network. Training is done by taking one step of the Levenberg-Marquardt algorithm, calculate the effecitve number of parameters by using the Gauss-Newton approximation to the Hessian matrix, then compute new estimates for the regularization parameters, then repeat until stopping criteria is met. Experiments are done on a number of networks with different types of activation function (sigmoid, ReLU, ELU, Leaky ReLU, etc.).

- **config.py: Set config option for logging.

- **conjugate_gradient_quadratic_function.py: Examples of the conjugate gradient descent algorithm for quadratic functions.

- **levenberg_marquadt_algorithm.py: Implement a neural network trained by the Levenberg Marquardt algorithm. Experiments are done on a number of networks with different types of activation function (sigmoid, ReLU, ELU, Leaky ReLU, etc.).

- **neural_network_regression.py: Implement a multi-layer feedforward neural network (the number of layers can be changed depending on the problem) for regression. A number of optimizers are supported: gradient descent, conjugate gradient descent, and the Adam algorithm.

- **optimizers.py: Implement several optimization algorithms, namely: gradient descent, conjugate gradient descent, and the Adam algorithm.

- **radial_basis_network.py: Implement the Radial basis function (RBF) network with the first layer orthogonal least square (OLS) algorithm, and the second layer trained using least mean square (LMS).

- **rnn.py: Implement the Recurrent Neural Network (RNN) architecture with support for arbitrary recurrent connection. Training is done by using the Backpropagation through time (BPTT) algorithm. A Keras-like API is supported to help users define layers and connections in the RNN network of their choice. 

- **rnn_maglev_experiments.py: An example of RNN to predict magnetic levitation. 

- **rnn_test.py: Some test function to test the correctness of the RNN implementation (including forward and BPTT).

- **rnn_utils.py: Some utility functions for RNN.

- **run.py: Run some test functions for neural networks using gradient descent, conjugate gradient descent and the Adam algorithm. Experiments are done on a number of networks with different types of activation function (sigmoid, ReLU, ELU, Leaky ReLU, etc.).

- **test_utility_functions.py: Test a number of utility functions

- **test_neural_network_regression.py: Run some test functions to check the correctness of the neural network implementation (both forward and backward pass).

- **utils.py: Utility function for neural network implementation (activation function, loss, etc.).

- **widrow_hoff_experiment.py: Implement a simple Perceptron network to run the experiment by Widrow and Hoff to recognize some digits. 

