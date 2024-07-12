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

