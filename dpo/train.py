# pass in the data (x , y_winner , y_loser)
#   here there is still quesitons, but i presume concat x,ys and then feed the models the slice
# compute the loss
# update the policy & policy optimizer
import jax
import jax.numpy as jnp
from flax import nnx
from .dpo import dpo_loss

@nnx.jit
def train_step(policy , reference , optimizer , batch):
    x , y_winner , y_loser = batch

    def loss_fn(policy):
        return dpo_loss(x , policy , reference , y_winner , y_loser)
    
    loss , grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(policy , grads)
    return loss

def train_epoch(policy , reference , optimizer , data):
    losses = []

    for batch in data:
        loss = train_step(policy , reference , optimizer , batch)
        losses.append(loss)

    return losses


if __name__ == "__main__":
    import optax
    from dpo.policy import PolicyModel
    from dpo.transformer import TransformerConfig

    # rngs
    rng = jax.random.PRNGKey(42)
    config = TransformerConfig()

    # models and optimizer
    policy = PolicyModel(config , rngs=nnx.Rngs(0))
    reference = PolicyModel(config , rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(policy , optax.adam(1e-3) , wrt=nnx.Param)

    # data
    BATCH , PROMPT_LEN , RESPONSE_LEN = 2 , 8 , 8

    rng_x , rng_winner , rng_loser = jax.random.split(rng , 3)
    x = jax.random.randint(rng_x , (BATCH,PROMPT_LEN) , 0 , config.VOCAB_SIZE)
    y_winner = jax.random.randint(rng_winner , (BATCH,RESPONSE_LEN) , 0 , config.VOCAB_SIZE)
    y_loser = jax.random.randint(rng_loser , (BATCH,RESPONSE_LEN) , 0 , config.VOCAB_SIZE)

    data = [(x , y_winner , y_loser)] * 5

    # training
    for epoch in range(3):
        losses = train_epoch(policy , reference , optimizer , data)
        print(f'epoch {epoch+1} | average loss = {sum(losses)/len(losses):.3f}')
