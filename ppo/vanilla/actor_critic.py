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
LEARNING_RATE = 3e-4
NUM_EPISODES = 1_000
ENVIRONMENT = 'CartPole-v1'

# Environment stuff
env , env_params = gymnax.make(ENVIRONMENT)

# Loss Funcitons
def actor_loss_fn(actor , obs , action , advantage):
    ...

def critic_loss_fn(critic , obs , target):
    return (critic(obs) - jax.lax.stop_gradient(target)) ** 2

# Training
def train(rngs):
    # Initializing models and optimizers
    actor = ActorModel(OBSERVATION_SIZE , HIDDEN , ACTION_SIZE , rngs=nnx.Rngs(0))
    critic = CriticModel(OBSERVATION_SIZE , HIDDEN , rngs=nnx.Rngs(1))

    optimizer_actor = nnx.Optimizer(actor , optax.adam(LEARNING_RATE) , wrt=nnx.Param)
    optimizer_critic = nnx.Optimizer(critic , optax.adam(LEARNING_RATE) , wrt=nnx.Param)

    # Splitting into the states
    graphdef_actor , state_actor = nnx.split(actor)
    graphdef_critic , state_critic = nnx.split(critic)
    graphdef_opt_a , state_opt_a = nnx.split(optimizer_actor)
    graphdef_opt_c , state_opt_c = nnx.split(optimizer_critic)

    def train_step():
        pass

    def run_episode():
        
        def body_fn(carry , _):
            pass

    @jax.jit
    def train_all_episodes():
        
        def scan_body(carry , _):
            pass