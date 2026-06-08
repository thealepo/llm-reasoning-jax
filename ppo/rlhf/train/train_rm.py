import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
from dataclasses import dataclass
import optax

from ..models.transformer import TransformerConfig
from ..models.reward_model import RewardModel

#===============================================================================================
# (input_ids_winner , input_ids_loser , mask_winner , mask_loser)
def bradley_terry_loss(y_winner , y_loser):
    return -jnp.log(jax.nn.sigmoid(y_winner - y_loser)).mean()

def train_step(state_model , state_optimizer , input_ids_winner , input_ids_loser , mask_winner , mask_loser):
    # Merging graphs with states
    model = nnx.merge(graphdef_rm , state_model)

    def loss_fn(params):
        rm = nnx.merge(graphdef_rm , params)
        score_winner , score_loser = rm(input_ids_winner , mask_winner) , rm(input_ids_loser , mask_loser)
        return bradley_terry_loss(score_winner , score_loser)
    
    # Gather losses and grads , create updates
    loss , grads = jax.value_and_grad(loss_fn)(nnx.state(model))
    model_updates , new_state_optimizer = state_optimizer.update(grads , state_model)
    new_state_model = optax.apply_updates(nnx.state(model) , model_updates)

    return (new_state_model , new_state_optimizer , loss)

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
    config = TransformerConfig()

    rng , rng_w , rng_l = jax.random.split(rng , 3)

    # FAKE DATA
    N_BATCHES , BATCH_SIZE , SEQ_LEN = 8 , 8 , 32
    input_ids_winners = jax.random.randint(rng_w , (N_BATCHES,BATCH_SIZE,SEQ_LEN) , 0 , config.VOCAB_SIZE)
    input_ids_losers = jax.random.randint(rng_l , (N_BATCHES,BATCH_SIZE,SEQ_LEN) , 0 , config.VOCAB_SIZE)
    mask_winners = jnp.ones((N_BATCHES,BATCH_SIZE,SEQ_LEN))
    mask_losers = jnp.ones((N_BATCHES,BATCH_SIZE,SEQ_LEN))

    reward_model = RewardModel(config , rngs=rngs)
    reward_optimizer = optax.adam(1e-3)
    state_rm_optimizer = reward_optimizer.init(nnx.state(reward_model))

    graphdef_rm , state_rm = nnx.split(reward_model)

    for epoch in range(10):
        t0 = time.time()
        final_carry , losses = train_epoch(state_rm , state_rm_optimizer , input_ids_winners , input_ids_losers , mask_winners , mask_losers)
        print(f'Epoch {epoch}, Mean Loss: {losses.mean():.4f}, Time: {time.time()-t0:.3f}s')
        state_rm , state_rm_optimizer = final_carry