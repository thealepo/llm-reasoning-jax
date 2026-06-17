# pass in the data (x , y_winner , y_loser)
#   here there is still quesitons, but i presume concat x,ys and then feed the models the slice
# compute the loss
# update the policy & policy optimizer
import jax
import jax.numpy as jnp
from flax import nnx
from dpo import dpo_loss

def train_step(policy , reference , optimizer , batch):
    x , y_winner , y_loser = batch

    def loss_fn(x , policy , reference , y_winner , y_loser):
        return dpo_loss(x , policy , reference , y_winner , y_loser)
    grad_fn = nnx.value_and_grad(loss_fn)
    
    loss , grads = grad_fn(x , policy , reference , y_winner , y_loser)
    optimizer.update(policy , grads)
    return loss

