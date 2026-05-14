import jax
import jax.numpy as np
import flax.nnx as nnx

class PolicyModel(nnx.Module):
    def __init__(self , input_size , hidden_size , output_size , * , rngs):
        self.fc1 = nnx.Linear(input_size , hidden_size , rngs=rngs)
        self.fc2 = nnx.Linear(hidden_size , output_size , rngs=rngs)

    def __call__(self , x):
        x = nnx.relu(self.fc1(x))
        logits = self.fc2(x)

        action = jax.random.categorical(rngs.params() , logits)

        log_probs = jax.nn.log_softmax(logits)
        log_probs_action = log_probs[jnp.arange(action.shape[0]) , action]

        return action , log_probs_action