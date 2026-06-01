import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnax

from model import ActorModel , CriticModel


# hyperparams
OBSERVATION_SIZE = 4
ACTION_SIZE = 2
HIDDEN = 128
GAMMA = 0.99
EPSILON = 0.2
LAMBDA = ...
K_EPOCHS = 4
N_STEPS = 128
LEARNING_RATE = 3e-4
NUM_EPISODES = 1_000
MAX_STEPS = 500
ENVIRONMENT = 'CartPole-v1'

# Environment stuff
env , env_params = gymnax.make(ENVIRONMENT)

# Loss Functions
def actor_loss_fn(actor , obs , actions , old_log_probs , advantages):
    # jax.vamp(function , in_axes , out_axes)
    logits = jax.vmap(actor)(obs)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = log_probs[jnp.arange(len(actions)) , actions]

    # Calculating the ratio from TRPO & PPO
    # (Policy's log probabilites / Policy's old log probabilities)
    ratio = jnp.exp(action_log_probs - jax.lax.stop_gradient(old_log_probs))

    # Calculating Surrogates
    cpi = ratio * advantages
    clipped = jnp.clip(ratio , 1 - EPSILON , 1 + EPSILON) * advantages

    # Policy Loss
    return -jnp.mean(jnp.minimum(cpi , clipped))


def critic_loss_fn(critic , obs , returns):
    values = jax.vmap(critic)(obs)  # obs is now (N_STEPS , 4) instead of (4,)
    return jnp.mean((values - jax.lax.stop_gradient(returns)) ** 2)

def compute_gae(rewards , values , dones):
    pass

def rollout(state_actor , state_critic , init_obs , init_env_state , rng):
    pass

def train_step(state_actor , state_critic , state_opt_a , state_opt_c , obs , actions , old_log_probs , advantages , returns):
    pass

def train_epoch(state_actor , state_critic , state_opt_a , state_opt_c , obs , actions , old_log_probs , advantages , returns):
    pass

@jax.jit
def train_all_episodes(state_actor , state_critic , state_opt_a , state_opt_c , rng):
    pass