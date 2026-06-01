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

def compute_gae(rewards , values , dones , next_value):
    # Delta (TD error). How much better or worse than expected
    next_values = jnp.append(values[1:] , next_value)
    deltas = rewards + GAMMA * next_values * (1 - dones) - values

    def scan_fn(last_advantage , delta_and_done):
        delta , done = delta_and_done
        advantage = delta + GAMMA * LAMBDA * (1 - done) * last_advantage
        return advantage , advantage

    # Reverse scan
    _ , advantages = jax.lax.scan(
        scan_fn,
        jnp.float32(0.0),
        (deltas , dones),
        reverse=True
    )

    returns = advantages + values
    return advantages , returns

def rollout(state_actor , state_critic , init_obs , init_env_state , rng):

    def body_fn(carry , _):
        obs , env_state , rng = carry
        rng , rng_action , rng_step = jax.random.split(rng , 3)

        # Sampling Actor
        actor = nnx.merge(graphdef_actor , state_actor)
        action , log_prob = actor.sample(obs , rng_action)

        # Value estimate from the Critic
        critic = nnx.merge(graphdef_critic , state_critic)
        value = critic(obs)

        # Environment step
        new_obs , new_env_state , reward , done , _ = env.step(
            rng_step , env_state , action , env_params
        )

        carry = (new_obs , new_env_state , rng)
        memory = (obs , action , reward , done , log_prob , value)
        return carry , memory

    # Initial carry
    init_carry = (init_obs , init_env_state , rng)

    # Full rollout + Memoruy and FInal carry
    final_carry , memory = jax.lax.scan(
        body_fn,
        init_carry,
        None,
        length=N_STEPS
    )
    final_obs , final_env_state , rng = final_carry
    obs , actions , rewards , dones , old_log_probs , values = memory

    # Value of the final state
    critic = nnx.merge(graphdef_critic , state_critic)
    next_value = critic(final_obs)

    # The advantages
    advantages , returns = compute_gae(rewards , values , dones , next_value)

    return (
        obs,
        actions,
        rewards,
        dones,
        old_log_probs,
        values,
        advantages,
        returns,
        final_obs,
        final_env_state,
        rng
    )

def train_step(state_actor , state_critic , state_opt_a , state_opt_c , obs , actions , old_log_probs , advantages , returns):
    # Update the weights and all for the actor and critic

    # Merging the splits
    actor = nnx.merge(graphdef_actor , state_actor)
    critic = nnx.merge(graphdef_critic , state_critic)
    optimizer_actor = nnx.merge(graphdef_opt_a , state_opt_a)
    optimizer_critic = nnx.merge(graphdef_opt_c , state_opt_c)

    # Losses and Gradient Updates
    a_loss , a_grads = nnx.value_and_grad(actor_loss_fn)(actor , obs , actions , advantages)
    optimizer_actor.update(actor , a_grads)

    c_loss , c_grads = nnx.value_and_grad(critic_loss_fn)(critic , obs , returns)
    optimizer_critic.update(critic , c_grads)

    return (nnx.state(actor) , nnx.state(critic) , nnx.state(optimizer_actor) , nnx.state(optimizer_critic) , a_loss , c_loss)

def train_epoch(state_actor , state_critic , state_opt_a , state_opt_c , obs , actions , old_log_probs , advantages , returns):
    pass

@jax.jit
def train_all_episodes(state_actor , state_critic , state_opt_a , state_opt_c , rng):
    pass