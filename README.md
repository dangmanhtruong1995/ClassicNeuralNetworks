# Neural networks

Implementations of a number of classic neural network and training algoritms. Most of the code is based on the descriptions in [1].

                [1] M. Hagan et al., Neural network design (2nd ed.), 2014. 

To install the necessary libraries, type the following (assuming you are using `conda`):

                conda create --name nndesign python=3.9
				
				pip install -r requirements.txt
				
- `adaptive_predictor.py`: Implement a simple Perceptron network. The network tries to predict based on results of two previous time steps.

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/adaptive_predictor_py_result.png "")

- `bayesian_regularized_neural_network.py`: Implement a Bayesian regularized neural network. Training is done by taking one step of the Levenberg-Marquardt algorithm, calculate the effecitve number of parameters by using the Gauss-Newton approximation to the Hessian matrix, then compute new estimates for the regularization parameters, then repeat until stopping criteria is met. Experiments are done on a number of networks with different types of activation function (sigmoid, ReLU, ELU, Leaky ReLU, etc.).

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Bayesian_1.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Bayesian_8.png "")

- `config.py`: Set config option for logging.

- `conjugate_gradient_quadratic_function.py`: Examples of the conjugate gradient descent algorithm for quadratic functions.

- `levenberg_marquadt_algorithm.py`: Implement a neural network trained by the Levenberg Marquardt algorithm. Experiments are done on a number of networks with different types of activation function (sigmoid, ReLU, ELU, Leaky ReLU, etc.).

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Levenberg_Marquadt_1.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Levenberg_Marquadt_2.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Levenberg_Marquadt_3.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Levenberg_Marquadt_4.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Levenberg_Marquadt_5.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Levenberg_Marquadt_6.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Levenberg_Marquadt_7.png "")

- `neural_network_regression.py`: Implement a multi-layer feedforward neural network (the number of layers can be changed depending on the problem) for regression. A number of optimizers are supported: gradient descent, conjugate gradient descent, and the Adam algorithm.

- `optimizers.py`: Implement several optimization algorithms, namely: gradient descent, conjugate gradient descent, and the Adam algorithm.

- `radial_basis_network.py`: Implement the Radial basis function (RBF) network with the first layer orthogonal least square (OLS) algorithm, and the second layer trained using least mean square (LMS).

- `rnn.py`: Implement the Recurrent Neural Network (RNN) architecture with support for arbitrary recurrent connection. Training is done by using the Backpropagation through time (BPTT) algorithm. A Keras-like API is supported to help users define layers and connections in the RNN network of their choice. 

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/RNN_equation.png "")

- `rnn_maglev_experiments.py`: An example of RNN to predict magnetic levitation. Below are predictions and loss after 100 and 500 epochs.

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/RNN_maglev_result_100_epoch.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/RNN_maglev_result_500_epoch.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/RNN_maglev_loss_100_epoch.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/RNN_maglev_loss_500_epoch.png "")

- `rnn_test.py`: Some test function to test the correctness of the RNN implementation (including forward and BPTT).

- `rnn_utils.py`: Some utility functions for RNN.

- `run.py`: Run some test functions for neural networks using gradient descent, conjugate gradient descent and the Adam algorithm. Experiments are done on a number of networks with different types of activation function (sigmoid, ReLU, ELU, Leaky ReLU, etc.). The results below are by: gradient descent (1st figure), conjugate gradient descent (2nd figure), and the Adam algorithm (3rd and 4th figure).

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Two_layer_net_1-5-1_steepest_descent_lr_0.1_100000_epochs.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Conjugate_gradient_Polak_Ribiere.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Adam_2000_epoch_lol.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/Adam_5000_epoch_lol.png "")

- `test_utility_functions.py`: Test a number of utility functions

- `test_neural_network_regression.py`: Run some test functions to check the correctness of the neural network implementation (both forward and backward pass).

- `utils.py`: Utility function for neural network implementation (activation function, loss, etc.).

- `widrow_hoff_experiment.py`: Implement a simple Perceptron network to run the experiment by Widrow and Hoff to recognize some digits. I also tried to change activation to Leaky ReLU.

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/widrow_hoff_experiment_result.png "")

![](https://github.com/dangmanhtruong1995/ClassicNeuralNetworks/blob/master/figures/widrow_hoff_experiment_result_leaky_relu.png "")

