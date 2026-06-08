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

def train_step(graphdefs , state_model , state_optimizer , input_ids_winner , input_ids_loser , mask_winner , mask_loser):
    graphdef_rm , optimizer_reward = graphdefs

    # Merging graphs with states
    model = nnx.merge(graphdef_rm , state_model)

    def loss_fn(params):
        rm = nnx.merge(graphdef_rm , params)
        score_winner , score_loser = rm(input_ids_winner , mask_winner) , rm(input_ids_loser , mask_loser)
        return bradley_terry_loss(score_winner , score_loser)
    
    # Gather losses and grads , create updates
    loss , grads = jax.value_and_grad(loss_fn)(nnx.state(model))
    model_updates , new_state_optimizer = optimizer_reward.update(grads , state_optimizer)
    new_state_model = optax.apply_updates(nnx.state(model) , model_updates)

    return new_state_model , new_state_optimizer , loss

def train_epoch(graphdefs , state_model , state_optimizer , input_ids_winners , input_ids_losers , mask_winners , mask_losers):

    def body_fn(carry , batch):
        state_model , state_optimizer = carry
        input_ids_winner , input_ids_loser , mask_winner , mask_loser = batch

        state_model , state_optimizer , loss = train_step(
            graphdefs,
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
train_epoch = jax.jit(train_epoch , static_argnames=('graphdefs',))
#================================================================================================

def train(graphdefs , state_model , state_optimizer , data_winners , data_losers , mask_winners , mask_losers , n_epochs=10):
    loss_history = []
    train_start = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        state_model , state_optimizer , losses = train_epoch(
            graphdefs , state_model , state_optimizer , 
            data_winners , data_losers , mask_winners , mask_losers
        )
        losses.block_until_ready()

        epoch_time = time.time() - epoch_start
        mean_loss = float(losses.mean())
        loss_history.append(mean_loss)

        print(
            f"epoch [{epoch+1}/{n_epochs}] "
            f"loss: {mean_loss:.4f} "
            f"time: {epoch_time:.2f}s"
        )
        if epoch == 0:
            print(f"  (first epoch includes JIT compile time)")

    total_time = time.time() - train_start
    print(f'Total training time: {total_time:.2f}s')

    return state_model , state_optimizer , loss_history

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
    graphdef_rm , state_rm = nnx.split(reward_model)
    optimizer_reward = optax.adam(1e-3)
    state_opt_rm = optimizer_reward.init(nnx.state(reward_model))

    graphdefs = (graphdef_rm , optimizer_reward)

    for epoch in range(10):
        t0 = time.time()
        final_carry , losses = train_epoch(graphdefs , state_rm , state_opt_rm , input_ids_winners , input_ids_losers , mask_winners , mask_losers)
        print(f'Epoch {epoch}, Mean Loss: {losses.mean():.4f}, Time: {time.time()-t0:.3f}s')
        state_rm , state_rm_optimizer = final_carry