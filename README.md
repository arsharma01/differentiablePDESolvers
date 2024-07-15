# Differential Stochastic PDE Solver
From heat sinks in computers to airplane wings, Partial Differential Equations (PDEs) are used everywhere to demarcate the constraints of physical systems. There has been a recent breakthrough in solving PDEs using [stochastic methods](https://cs.dartmouth.edu/~wjarosz/publications/sawhneyseyb22gridfree.html). Our primary goal is to makes these processes differentiable to enable us to solve PDE constrained optimization problems using gradient based optimization.

# Description of Previous Work
Our primary source paper is [this paper from Sawhney and Crane](https://cs.dartmouth.edu/~wjarosz/publications/sawhneyseyb22gridfree.html) where we have a stochastic method to solve PDEs of the form 

$$
CE(p,y)=\left\{
\begin{array}{ll}
-\log(p) &\text{if }y=1 \\ 
-\log(1-p) &\text{otherwise}.
\end{array} 
\right.
$$

<div>
$$
\begin{cases}\tag{1}
\nabla\cdot(\alpha\nabla u) + \omega \cdot \nabla u - \sigma u = f & \text{ on } \Omega \\
u = g & \text{ on } \partial \Omega
\end{cases}
$$
</div>

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true
    }
  });
</script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


where $\alpha,\omega \, \text{and} \,\sigma$ are spatially varying coefficients. 

This paper [Yilmazer et al.(2022)](https://arxiv.org/pdf/2208.02114) shows how one can calculate the derivate with respect to a parameter $\pi$ with the condition that the **domain remains unchanged**.

A recent paper by [Yu et at. (2024)](https://www.shuangz.com/projects/diff-wos-sg24/diff-wos-sg24.pdf) shows how we can calculate the derivative to the solution to our PDE w.r.t. a parameter $\theta$ which changes also the boundary shape. However, **this has only been done for the Poisson problem** where our condition is that: $$ \Delta u = -f \text{ on } \Omega $$ and $$ u = g \text { on } \partial \Omega $$

We will be working on extending this result to the general case of equation (1).
# Contents
We provide a module ``dirichlet_solver.py`` which solves PDEs with dirichlet boundary conditions: 
$$
\begin{cases}
 \Delta u = 0 & \text{ on } \Omega \\
u = g & \text{ on } \partial \Omega
\end{cases}
$$
This uses the Walk On Spheres (WoS) method and also solves for the gradient of the solution. 

We are working on providing modules to solve PDEs with the general conditions (1) and once we have worked out the math for differentiating w.r.t. a parameter that changes the domain shape, we will also supplement code to solve for that.  


# Example README

Here is a case statement in LaTeX:

![cases](https://latex.codecogs.com/png.latex?%5Cbegin%7Bcases%7D%20a%20%26%20%5Ctext%7Bif%20%7D%20b%20%5C%5C%20c%20%26%20%5Ctext%7Bif%20%7D%20d%20%5Cend%7Bcases%7D)
