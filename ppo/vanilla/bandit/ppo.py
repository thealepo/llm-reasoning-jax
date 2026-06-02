import jax
import jax.numpy as jnp
from flax import nnx
import optax

from model import ActorModel


# Hyperparams
NUM_ARMS = 4
HIDDEN = 128
EPSILON = 0.2
K_EPOCHS = 4
N_STEPS = 128
LEARNING_RATE = 3e-4
NUM_EPISODES = 1_000

# Env
TRUE_REWARD_PROB = jnp.array([0.2 , 0.3 , 0.9 , 0.2])
OBS = jnp.zeros(NUM_ARMS)

# Bandit thing
def pull(action , rng):
    return jax.random.bernoulli(rng , TRUE_REWARD_PROB[action]).astype(jnp.float32)

# Losses
def actor_loss_fn(actor , actions , old_log_probs , advantages):
    logits = jax.vmap(actor)(OBS)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = log_probs[jnp.arange(N_STEPS) , actions]

    ratio = jnp.exp(action_log_probs - jax.lax.stop_gradient(old_log_probs))

    cpi = ratio * jax.lax.stop_gradient(advantages)
    clipped = jnp.clip(ratio , 1 - EPSILON , 1 + EPSILON) * jax.lax.stop_gradient(advantages)

    return -jnp.mean(jnp.minimum(cpi , clipped))


def rollout(state_actor , rng):

    def body_fn(carry , _):
        rng = carry
        rng , rng_action , rng_pull = jax.random.split(rng , 3)

        actor = nnx.merge(graphdef_actor , state_actor)
        action , log_prob = actor.sample(OBS , rng_action)

        reward = pull(action , rng_pull)

        return rng , (action , reward , log_prob)

    final_rng , (actions , rewards , old_log_probs) = jax.lax.scan(body_fn , rng , None , length=N_STEPS)

    advantages = rewards - rewards.mean()
    return actions , rewards , old_log_probs , advantages , final_rng

def train_step(state_actor , state_opt_a , actions , old_log_probs , advantages):
    actor = nnx.merge(graphdef_actor , state_actor)
    optimizer_actor = nnx.merge(graphdef_opt_a , state_opt_a)

    def a_loss(params):
        a = nnx.merge(actor , params)
        return actor_loss_fn(a , actions , old_log_probs , advantages)

    a_loss_val , a_grads = jax.value_and_grad(a_loss)(nnx.state(actor))
    optimizer_actor.update(actor , a_grads)

    return nnx.state(actor) , nnx.state(optimizer_actor) , a_loss_val

def train_epoch(state_actor , state_opt_a , old_log_probs , advantages):

    def body_fn(i , carry):
        state_actor , state_opt_a = carry
        state_actor , state_opt_a , _ = train_step(
            state_actor , state_opt_a , actions , old_log_probs , advantages
        )
        return state_actor , state_opt_a

    carry = (state_actor , state_opt_a)
    return jax.lax.fori_loop(0 , K_EPOCHS , body_fn , carry)

@jax.jit
def train_all_episodes(state_actor , state_opt_a , rng):

    def scan_fn(carry , _):
        state_actor , state_opt_a , rng = carry

        actions , rewards , old_log_probs , advantages , rng = rollout(state_actor , rng)

        state_actor , state_opt_a = train_epoch(
            state_actor , state_opt_a , old_log_probs , advantages
        )

        carry = (state_actor , state_opt_a , rng)
        return carry , rewards.sum()

    init_carry = (state_actor , state_opt_a , rng)
    final_carry , episode_rewards = jax.lax.scan(
        scan_fn , init_carry , None , length=NUM_EPISODES
    )

    return final_carry , episode_rewards

# entr
def train(rng):
    actor = ActorModel(NUM_ARMS , HIDDEN , NUM_ARMS , rngs=nnx.Rngs(42))
    optimizer_actor = nnx.Optimizer(actor , optax.adam(LEARNING_RATE) , wrt=nnx.Param)

    global graphdef_actor , graphdef_opt_a
    graphdef_actor , state_actor = nnx.split(actor)
    graphdef_opt_a , state_opt_a = nnx.split(optimizer_actor)

    return train_all_episodes(state_actor , state_opt_a , rng)