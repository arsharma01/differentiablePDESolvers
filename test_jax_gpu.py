import jax
import jax.numpy as jnp

# Check if JAX is using GPU
from jax.lib import xla_bridge
print("JAX is using:", xla_bridge.get_backend().platform)

# Run a simple computation
x = jnp.array([1.0, 2.0, 3.0])
print(x)
