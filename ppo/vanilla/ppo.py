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
def actor_loss_fn(actor , obs , action , advantage):
    log_probs = jax.nn.log_softmax(actor(obs))
    return -log_probs[action] * jax.lax.stop_gradient(advantage)

def critic_loss_fn(critic , obs , target):
    return (critic(obs) - jax.lax.stop_gradient(target)) ** 2

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