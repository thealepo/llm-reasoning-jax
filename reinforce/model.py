import jax
import jax.numpy as jnp
import flax.nnx as nnx

class PolicyModel(nnx.Module):
    def __init__(self , input_size , hidden_size , output_size , * , rngs):
        self.fc1 = nnx.Linear(input_size , hidden_size , rngs=rngs)
        self.fc2 = nnx.Linear(hidden_size , output_size , rngs=rngs)

    def __call__(self , x):
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        return x

    def sample(self , x , key):
        logits = self(x)
        action = jax.random.categorical(key , logits)
        log_prob = jax.nn.log_softmax(logits)[action]
        return action , log_prob