import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnax

from .model import ActorModel , CriticModel


# hyperparams
OBSERVATION_SIZE = 4
ACTION_SIZE = 2
HIDDEN = 128
GAMMA = 0.99
EPSILON = 0.2
LAMBDA = 0.95
K_EPOCHS = 4
N_STEPS = 128
LEARNING_RATE = 3e-4
NUM_EPISODES = 1_000
# MAX_STEPS = 500
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
    cpi = ratio * jax.lax.stop_gradient(advantages)
    clipped = jnp.clip(ratio , 1 - EPSILON , 1 + EPSILON) * jax.lax.stop_gradient(advantages)

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

    # Trace-level loss function correction
    def a_loss(params):
        a = nnx.merge(graphdef_actor , params)
        return actor_loss_fn(a , obs , actions , old_log_probs , advantages)
    def c_loss(params):
        c = nnx.merge(graphdef_critic , params)
        return critic_loss_fn(c , obs , returns)

    # Gathering losses, grads, and updating weights
    a_loss_val , a_grads = jax.value_and_grad(a_loss)(nnx.state(actor))
    c_loss_val , c_grads = jax.value_and_grad(c_loss)(nnx.state(critic))
    optimizer_actor.update(actor, a_grads)
    optimizer_critic.update(critic, c_grads)

    return (nnx.state(actor) , nnx.state(critic) , nnx.state(optimizer_actor) , nnx.state(optimizer_critic) , actor_loss_val , critic_loss_val)


def train_epoch(state_actor , state_critic , state_opt_a , state_opt_c , obs , actions , old_log_probs , advantages , returns):

    def body_fn(i , carry):
        state_actor , state_critic , state_opt_a , state_opt_c = carry

        # Take a training step
        state_actor , state_critic , state_opt_a , state_opt_c , a_loss , c_loss = train_step(
            state_actor,
            state_critic,
            state_opt_a,
            state_opt_c,
            obs,
            actions,
            old_log_probs,
            advantages,
            returns
        )

        return (state_actor , state_critic , state_opt_a , state_opt_c)

    init_carry = (state_actor , state_critic , state_opt_a , state_opt_c)
    final_carry = jax.lax.fori_loop(0 , K_EPOCHS , body_fn , init_carry)

    return final_carry


@jax.jit
def train_all_episodes(state_actor , state_critic , state_opt_a , state_opt_c , rng):
    
    def scan_fn(carry , _):
        state_actor , state_critic , state_opt_a , state_opt_c , obs , env_state , rng = carry

        # Rollout (collect data)
        (obs_batch , actions , rewards , dones , old_log_probs , values , advantages , returns , next_obs , next_env_state , rng) = rollout(
            state_actor , state_critic , obs , env_state , rng
        )

        # Update the weights per epoch
        state_actor , state_critic , state_opt_a , state_opt_c = train_epoch(
            state_actor , state_critic , state_opt_a , state_opt_c , obs_batch , actions , old_log_probs , advantages , returns
        )

        # Carry
        carry = (state_actor , state_critic , state_opt_a , state_opt_c , next_obs , next_env_state , rng)
        return carry , rewards.sum()

    # Initiale environment reset
    rng , rng_reset = jax.random.split(rng)
    init_obs , init_env_state = env.reset(rng_reset , env_params)

    init_carry = (
        state_actor,
        state_critic,
        state_opt_a,
        state_opt_c,
        init_obs,
        init_env_state,
        rng
    )

    final_carry , episode_rewards = jax.lax.scan(
        scan_fn , init_carry , None , length=NUM_EPISODES
    )
    return final_carry , episode_rewards


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    rng , actor_rng , critic_rng , train_rng = jax.random.split(rng , 4)

    actor = ActorModel(OBSERVATION_SIZE , HIDDEN , ACTION_SIZE , rngs=actor_rng)
    critic = CriticModel(OBSERVATION_SIZE , rngs=critic_rng)
    optimizer_actor = nnx.Optimizer(actor , optax.adam(LEARNING_RATE) , wrt=nnx.Param)
    optimizer_critic = nnx.Optimizer(critic , optax.adam(LEARNING_RATE) , wrt=nnx.Param)

    graphdef_actor , state_actor = nnx.split(actor)
    graphdef_critic , state_critic = nnx.split(critic)
    graphdef_opt_a , state_opt_a = nnx.split(optimizer_actor)
    graphdef_opt_c , state_opt_c = nnx.split(optimizer_critic)

    _ , episode_rewards = train_all_episodes(
        state_actor , state_critic , state_opt_a , state_opt_c , rng=train_rng
    )

    print(f'First episode reward: {episode_rewards[0]:.1f}')
    print(f'Last episode reward: {episode_rewards[-1]:.1f}')
