# module to solve any dirichlet problem and relevant gradients
# using the Walk On Spheres (WoS) Method

# import statements
import jax
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import plotly.graph_objs as go
import jax.numpy as jnp


# our 'h' function on the boundary
#### change as necessary ####
def boundary_function(x,y,z):
    # for sufficiently small eps, we can make this approximation
    index = jnp.argmin( jnp.array([x,y,1-x,1-y]) )
    if index == 1:
        return z
    else:
        return 0
    
# the gradients of our 'h' function on the boundary w.r.t. parameter z
#### change as necessary ####
def boundary_function_grad(x,y,z):
    # for sufficiently small eps, we can make this approximation
    index = jnp.argmin( jnp.array([x,y,1-x,1-y]) )
    if index == 1:
        return 1
    else:
        return 0
    
# given a point (x,y) returns the radius of the largest circle centered
# at (x,y) and lies within our domain
#### change as necessary ####
def largest_radius(x,y):
    return min(x,y,1-x,1-y)

# solves for u at specified x,y withing the domain using Walk On Spheres
# returns u(x,y) and du(x,y)/ dz
def solve(x = 1/4,y = 1/4, z=0, eps = 1e-4,max_iters = 1000):   
    running_function_value = 0
    running_gradient_value = 0
    # walk on spheres
    for _ in range(max_iters):
        x_new,y_new = x,y 
        r_max = largest_radius(x,y)
        # check point is outisde eps distance from boundary
        while  r_max> eps:
            # pick random point on circle 
            theta = np.random.uniform(0, 2 * jnp.pi)
            x_new, y_new = x_new + jnp.cos(theta) * r_max, y_new + jnp.sin(theta) * r_max
            # calculate the largest circle that will fit in domain
            r_max = largest_radius(x_new, y_new)

        running_function_value += boundary_function( x_new, y_new, z)
        running_gradient_value += boundary_function_grad( x_new, y_new, z)


    # taking respective means
    return running_function_value/max_iters, running_gradient_value/max_iters


# loss function for gradient descent
# return function value and gradient
def loss_function(x = 1/4,y = 1/4, z=10, target_value = 15, eps = 1e-4,max_iters = 500):
    function_value, gradient = solve(x = 1/4,y = 1/4, z=z, eps = 1e-4,max_iters = 500)
    gradient_of_loss_funct = -2*gradient*(target_value - function_value)

    return (function_value - target_value)**2 , gradient_of_loss_funct

# solves for an inverse problem using gradient descent
def solve_inverse_grad_dsc( x =1/4, y =1/4, z_init = 30, target_value = 15, loss_function, learning_rate, num_iterations,eps = 1e-4,max_iters_for_solve = 500, progress_report = False) :
    z = z_init
    for i in range(num_iterations):
        # Compute the loss function value and gradient w.r.t. z
        loss_value, gradient = loss_function(x,y, z, target_value, eps,max_iters_for_solve)
        
        if loss_value<0.01:
            return z
        
        # Update parameters
        z = z - learning_rate * gradient
        
        # Optionally print progress
        if progress_report and i%20 == 0:
            print(f"Iteration {i+1}: loss = {loss_value}, z = {z}")
    
    return z

# consider the unit square with temperature z at the x-axis. 
# sequential search for unit square
def seq_search(x = 1/4, y =1/4, target_temp = 15, heater_low = 0, heater_high= 100, error_min = 0.1, prog_report = False):
    errors_low = []
    errors_high = []
    count = 0
    # initial
    temp_low = solve(x,y,z = heater_low)[0]
    low_error =np.abs(temp_low - target_temp) 
    errors_low.append(low_error)

    temp_high = solve(z = heater_high)[0]
    high_error =np.abs(temp_high - target_temp) 
    errors_high.append(high_error)

    if target_temp<temp_low or target_temp>temp_high:
        print(f"Initial heater settings are insufficent")
        print(f"Initial temperatures are: {temp_low} and {temp_high}")
        print("Please readjust")
        return
    #success
    if low_error< error_min:
        return heater_low       
    if high_error< error_min:
        return heater_high
    
    count+=1
    
    # repeat
    while count <=2 or jnp.abs(errors_low[-1] - errors_low[-2])/errors_low[-1] > error_min or jnp.abs(errors_low[-1] - errors_low[-2])/errors_low[-1] > error_min:
           
        # if closer to lower temperature
        if low_error < high_error:
            heater_high = (heater_high+heater_low)/2
            temp_high = solve(z = heater_high)[0]
            high_error =np.abs(temp_high - target_temp) 
            errors_high.append(high_error)
            if high_error< error_min:
                return heater_high
        # if closer to higher temperature
        else:
            heater_low = (heater_high+heater_low)/2
            temp_low = solve(z = heater_low)[0]
            low_error =np.abs(temp_low - target_temp) 
            errors_low.append(low_error)
            if low_error< error_min:
                return heater_low
            
        if prog_report:
            count += 1
            if count%10 == 0:
                print(f"Iter: {count}, error_low: {low_error}, error_low:{high_error}")
                print(f"temp_low: {heater_low}, temp_high:{heater_high}")

            print(f"Iter: {count}, error_low: {low_error}, error_low:{high_error}")
            print(f"temp_low: {temp_low}, temp_high:{temp_high}")
            print(f"heater_low: {heater_low}, heater_high:{heater_high}")


    if prog_report:
        print("we reach final step")
    if errors_low[-1] < errors_high[-1]:
        return heater_low
    else:
        return heater_high
        