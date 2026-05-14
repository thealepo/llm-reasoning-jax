import jax
import jax.numpy as np
import flax.nnx as nnx

class PolicyModel(nnx.Module):
    def __init__(self , rngs):
        self.model = nnx.Sequential(
            nnx.Linear(1 , 32 , rngs=rngs),
            nnx.Linear(32 , 64 , rngs=rngs),
        )
    
    def __call__(self , x):
        return self.model(x)