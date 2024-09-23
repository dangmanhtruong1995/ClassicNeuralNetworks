import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import math
import scipy
import matplotlib.pyplot as plt
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

def perform_interval_reduction(func, eps):
    print("Perform interval reduction")
    
    step = 0
    F_prev_2 = func(step*eps)
    print("Step: %d" % (step))
    print("Function value: %f" % (F_prev_2))
    
    step = 1    
    F_prev_1 = func(step*eps)
    print("Step: %d" % (step))
    print("Function value: %f" % (F_prev_1))   
    
    step *= 2
    F_curr = func(step*eps)
    print("Step: %d" % (step))
    print("Function value: %f" % (F_curr))
    
    if F_curr > F_prev_1:
        left_interval = (step / 4)*eps
        right_interval = step*eps
        return (left_interval, right_interval)
    
    while True:
        step *= 2
        F_new = func(step*eps)
        print("Step: %d" % (step))
        print("Function value: %f" % (F_new))
        
        if F_new > F_curr:
            left_interval = (step / 4)*eps
            right_interval = step*eps
            return (left_interval, right_interval)
        
        F_prev_2 = F_prev_1
        F_prev_1 = F_curr
        F_curr = F_new  

def perform_golden_section_search(func, left_interval, right_interval, tau, tol):
    a_curr = left_interval
    b_curr = right_interval
    
    c_curr = a_curr + (1-tau)*(b_curr-a_curr)
    d_curr = b_curr - (1-tau)*(b_curr-a_curr)
    # set_trace()
    Fc = func(c_curr)
    Fd = func(d_curr)
    
    print("Begin Golden section search")
    print("a = %f, b=%f, c=%f, d=%f" % (a_curr, b_curr, c_curr, d_curr))    
    print("Fc = %f" % (Fc))
    print("Fd = %f" % (Fd))
    
    iteration= 0
    while True:
        iteration += 1
        print("Iteration: %d" % (iteration)) 
        
        if Fc < Fd:
            b_curr = d_curr
            d_curr = c_curr            
            c_curr = a_curr + (1-tau)*(b_curr-a_curr)
            Fd = Fc
            Fc = func(c_curr)            
        else:
            a_curr = c_curr
            c_curr = d_curr
            d_curr = b_curr - (1-tau)*(b_curr-a_curr)
            Fc = Fd
            Fd = func(d_curr)
            
        print("a = %f, b=%f, c=%f, d=%f" % (a_curr, b_curr, c_curr, d_curr))    
        print("Fc = %f" % (Fc))
        print("Fd = %f" % (Fd))
        
        if (b_curr-a_curr) < tol:
            print("b-a smaller than tolerance level. Terminating")
            break
            
    opt_val =  (b_curr+a_curr) / 2.0      
    print("Result: %f" % (opt_val))
    print("")
     
    return opt_val

def perform_interval_search(func, eps=0.075, tau=0.618, tol=0.01):
    (left_interval, right_interval) = perform_interval_reduction(func, eps)
    print("Result by interval reduction:")
    print((left_interval, right_interval))

    alpha_opt = perform_golden_section_search(func, left_interval, right_interval, tau, tol)
    print("Learning rate found by golden section search: %f" % (alpha_opt))
    
    return alpha_opt
    

def conjugate_gradient_for_quadratic_function(A1, b1, c1, x0, n_epoch=2,
        method="fletcher_and_reeves", interval_search=False):
    # Perform conjugate gradient method for a quadratic function (1/2)*A1*xTx + b1*x + c1
    
    grad_func = lambda x_in: np.dot(A1, x_in)+b1
    grad_prev = None
    p_prev = None
    grad_prev_mult_grad_prev = None # For the fletcher_and_reeves method
    
    n_dim = np.shape(A1)[0]
    # set_trace()
    
    # temp = grad(np.array([0.8, -0.25]))
    # temp = grad(np.array([0.24, -0.37]))
    x_curr = deepcopy(x0)
    for epoch_idx in range(1, n_epoch+1):
        print("Epoch %d" % (epoch_idx))
        print("Current position:")
        print(x_curr)
    
        # Calculate gradient at current position
        grad = grad_func(x_curr)
        print("Gradient:")
        print(grad)
        
        # Calculate search direction
        if (epoch_idx == 1) or ((epoch_idx-1) % n_dim == 0):
            # First step is the same as with steepest descent
            # After every n_dim step the conjugate gradient algorithm is repeated
            p_curr = -grad
            grad_prev = None
            p_prev = None
            grad_prev_mult_grad_prev = None # For the fletcher_and_reeves method
        else:
            # Now is the conjugate gradient part
            print("Using method: %s" % (method))
            if method == "fletcher_and_reeves":
                grad_mult_grad = np.dot(grad.transpose(), grad)
                # denominator = np.dot(grad_prev.transpose(), grad_prev)
                if grad_prev_mult_grad_prev is None:
                    grad_prev_mult_grad_prev = np.dot(grad_prev.transpose(), grad_prev)
                beta = grad_mult_grad / grad_prev_mult_grad_prev
                print("Beta: %f" % (beta))
                p_curr = -grad + beta*p_prev                
                grad_prev_mult_grad_prev = deepcopy(grad_mult_grad) # For more efficient calculation
                
            elif method == "hestenes_and_stiefel":
                diff_grad_prev = grad - grad_prev
                diff_grad_prev_mult_grad = np.dot(diff_grad_prev, grad)
                diff_grad_prev_mult_p_prev = np.dot(diff_grad_prev, p_prev)                
                beta = diff_grad_prev_mult_grad / diff_grad_prev_mult_p_prev
                print("Beta: %f" % (beta))
                p_curr = -grad + beta*p_prev  
                
            elif method == "polak_and_ribiere":
                diff_grad_prev = grad - grad_prev
                diff_grad_prev_mult_grad = np.dot(diff_grad_prev, grad)
                grad_prev_mult_grad_prev = np.dot(grad_prev.transpose(), grad_prev)
                beta = diff_grad_prev_mult_grad / grad_prev_mult_grad_prev
                print("Beta: %f" % (beta))
                p_curr = -grad + beta*p_prev  
                pass               
            pass

        print("p_curr: ")
        print(p_curr)
        
        # Calculate learning rate
        if interval_search is False:
            # Apply line search for the quadratic function
            numerator = np.dot(grad.transpose(), p_curr)
            denominator = np.dot(np.dot(p_curr, A1), p_curr)
            alpha_curr = (-1)*numerator / denominator
        else:
            # Perform interval search using interval reduction and golden section search 
            quadratic_func = lambda x_in: 0.5*(np.dot(np.dot(np.transpose(x_in), A1), x_in)) + np.dot(b1, x_in) + c1
            interval_func = lambda alpha: quadratic_func(x_curr+alpha*p_curr)
            # set_trace()
            alpha_curr = perform_interval_search(interval_func)            

        print("Learning rate alpha: %f" % (alpha_curr))
 
        # Step
        x_new = x_curr + alpha_curr*p_curr
        print("New step to: ")
        print(x_new)
        print("")
        
        # Prepare for the next epoch
        p_prev = deepcopy(p_curr)
        grad_prev = deepcopy(grad)
        x_curr = deepcopy(x_new)

        
    return x_curr

def main():
    A1 = np.array([
        [2, 1],
        [1, 2],
    ])
    b1 = np.array(
        [0, 0]
    )
    c1 = 0
    x0 = np.array(
        [0.8, -0.25]
    )
    n_epoch = 10
    
    
    # A1 = np.array([
        # [10, 2],
        # [ 2,  4],
    # ])
    # b1 = np.array(
        # [-2, -1]
    # )
    # c1 = 0.25
    # x0 = np.array(
        # [1,  1]
    # )
    # n_epoch = 2
    
    x_opt = conjugate_gradient_for_quadratic_function(A1, b1, c1, x0, n_epoch=n_epoch,
        # method="fletcher_and_reeves",
        # method="polak_and_ribiere",
        method="hestenes_and_stiefel",
        interval_search=True,
        )
    
    # set_trace()

if __name__ == "__main__":
    main()
