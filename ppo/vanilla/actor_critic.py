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

    def train_step(state_actor , state_critic , state_opt_a , state_opt_c , obs , action , reward , new_obs , done):
        # Merging the splits
        actor = nnx.merge(graphdef_actor , state_actor)
        critic = nnx.merge(graphdef_critic , state_critic)
        optimizer_actor = nnx.merge(graphdef_opt_a , state_opt_a)
        optimizer_critic = nnx.merge(graphdef_opt_c , state_opt_c)

        # Calculating the advantage
        value = critic(obs)
        next_value = critic(new_obs)
        target = reward + GAMMA * next_value * (1.0 - done)
        advantage = target - value

        # Losses and Gradients
        a_loss , a_grads = nnx.value_and_grad(actor_loss_fn)(actor , obs , action , advantage)
        optimizer_actor.update(actor , a_grads)

        c_loss , c_grads = nnx.value_and_grad(critic_loss_fn)(critic , obs , target)
        optimizer_critic.update(critic , c_grads)

        # Return
        return (nnx.state(actor) , nnx.state(critic) , nnx.state(optimizer_actor) , nnx.state(optimizer_critic) , a_loss , c_loss)

    def run_episode():
        
        def body_fn(carry , _):
            pass

    @jax.jit
    def train_all_episodes():
        
        def scan_body(carry , _):
            pass