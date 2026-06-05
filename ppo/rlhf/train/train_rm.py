import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
from dataclasses import dataclass
import optax

#===============================================================================================
# (input_ids_winner , input_ids_loser , mask_winner , mask_loser)
def bradley_terry_loss(y_winner , y_loser):
    return -jnp.log(jax.nn.sigmoid(y_winner - y_loser)).mean()

def train_step(state_model , state_optimizer , input_ids_winner , input_ids_loser , mask_winner , mask_loser):
    model = nnx.merge(graphdef_rm , state_model)
    optimizer = nnx.merge(graphdef_rm_optimizer , state_optimizer)

    def loss_fn(params):
        rm = nnx.merge(graphdef_rm , params)
        score_winner , score_loser = rm(input_ids_winner , mask_winner) , rm(input_ids_loser , mask_loser)
        return bradley_terry_loss(score_winner , score_loser)
    loss_val , grads = jax.value_and_grad(loss_fn)(nnx.state(model))
    optimizer.update(model , grads)

    return (nnx.state(model) , nnx.state(optimizer) , loss_val)

@jax.jit
def train_epoch(state_model , state_optimizer , input_ids_winners , input_ids_losers , mask_winners , mask_losers):

    def body_fn(carry , batch):
        state_model , state_optimizer = carry
        input_ids_winner , input_ids_loser , mask_winner , mask_loser = batch

        state_model , state_optimizer , loss = train_step(
            state_model,
            state_optimizer,
            input_ids_winner,
            input_ids_loser,
            mask_winner,
            mask_loser
        )
        return (state_model , state_optimizer) , loss

    init_carry = (state_model , state_optimizer)
    batches = (input_ids_winners , input_ids_losers , mask_winners , mask_losers)
    final_carry , losses = jax.lax.scan(
        body_fn , init_carry , batches
    )
    return final_carry , losses
#================================================================================================

if __name__ == "__main__":
    import time

    rng = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(rng)
    config = Config()

    rng , rng_w , rng_l = jax.random.split(rng , 3)

    # FAKE DATA
    N_BATCHES , BATCH_SIZE , SEQ_LEN = 8 , 8 , 32
    input_ids_winners = jax.random.randint(rng_w , (N_BATCHES,BATCH_SIZE,SEQ_LEN) , 0 , config.VOCAB_SIZE)
    input_ids_losers = jax.random.randint(rng_l , (N_BATCHES,BATCH_SIZE,SEQ_LEN) , 0 , config.VOCAB_SIZE)
    mask_winners = jnp.ones((N_BATCHES,BATCH_SIZE,SEQ_LEN))
    mask_losers = jnp.ones((N_BATCHES,BATCH_SIZE,SEQ_LEN))

    reward_model = Transformer(config , rngs=rngs)
    reward_optimizer = nnx.Optimizer(reward_model , optax.adam(1e-3) , wrt=nnx.Param)

    graphdef_rm , state_rm = nnx.split(reward_model)
    graphdef_rm_optimizer , state_rm_optimizer = nnx.split(reward_optimizer)

    for epoch in range(10):
        t0 = time.time()
        final_carry , losses = train_epoch(state_rm , state_rm_optimizer , input_ids_winners , input_ids_losers , mask_winners , mask_losers)
        print(f'Epoch {epoch}, Mean Loss: {losses.mean():.4f}, Time: {time.time()-t0:.3f}s')
        state_rm , state_rm_optimizer = final_carry