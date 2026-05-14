import jax
import jax.numpy as np
import flax.nnx as nnx

class PolicyModel(nnx.Module):
    def __init__(self , input_size , hidden_size , output_size , * , rngs):
        self.fc1 = nnx.Linear(input_size , hidden_size , rngs=rngs)
        self.fc2 = nnx.Linear(hidden_size , output_size , rngs=rngs)

    def __call__(self , x):
        x = self.fc1(x)
        x = nnx.relu(x)
        logits = self.fc2(x)

        probs = jax.nn.softmax(logits)
        action = jax.random.categorical(rngs.params() , logits)

        log_probs = jax.nn.log_softmax(logits)

        return