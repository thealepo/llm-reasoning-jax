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
NUM_ROLLOUTS = 1_000
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
    policy_loss = -jnp.mean(jnp.minimum(cpi , clipped))
    return policy_loss


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

def compute_mean_episode_reward(rewards , dones):
    
    def body_fn(carry , rd):
        rewards , dones = rd
        carry += rewards

        episode_return = jnp.where(dones , carry , 0.0)
        carry = jnp.where(dones , 0.0 , carry)
        return carry , episode_return

    _ , episode_returns = jax.lax.scan(body_fn , 0.0 , (rewards , dones))

    num_episodes = dones.sum()

    return jnp.where(
        num_episodes > 0,
        episode_returns.sum() / num_episodes,
        rewards.sum()
    )

def rollout(state_actor , state_critic , init_obs , init_env_state , rng):

    def body_fn(carry , _):
        obs , env_state , rng = carry
        rng , rng_action , rng_step , rng_reset = jax.random.split(rng , 4)

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

        # Environment Reset
        reset_obs , reset_env_state = env.reset(rng_reset , env_params)
        new_obs = jnp.where(done , reset_obs , new_obs)
        new_env_state = jax.tree.map(
            lambda r , s: jnp.where(done , r , s),
            reset_env_state , new_env_state
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

    return (nnx.state(actor) , nnx.state(critic) , nnx.state(optimizer_actor) , nnx.state(optimizer_critic) , a_loss_val , c_loss_val)


def train_epoch(state_actor , state_critic , state_opt_a , state_opt_c , obs , actions , old_log_probs , advantages , returns):

    def body_fn(i , carry):
        state_actor , state_critic , state_opt_a , state_opt_c = carry

        # Take a training step
        state_actor , state_critic , state_opt_a , state_opt_c , _ , _ = train_step(
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

        average_episode_reward = compute_mean_episode_reward(rewards, dones)

        return carry , average_episode_reward

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


# Testing the speed and validity of implementation
if __name__ == "__main__":
    import time

    rng = jax.random.PRNGKey(42)
    rng , rng_train = jax.random.split(rng , 2)

    # Initializing the models and optimizers
    actor = ActorModel(OBSERVATION_SIZE , HIDDEN , ACTION_SIZE , rngs=rngs)
    critic = CriticModel(OBSERVATION_SIZE , HIDDEN , rngs=rngs)
    optimizer_actor = nnx.Optimizer(actor , optax.adam(LEARNING_RATE) , wrt=nnx.Param)
    optimizer_critic = nnx.Optimizer(critic , optax.adam(LEARNING_RATE) , wrt=nnx.Param)

    # Splitting the architecture from the parameters
    graphdef_actor , state_actor = nnx.split(actor)
    graphdef_critic , state_critic = nnx.split(critic)
    graphdef_opt_a , state_opt_a = nnx.split(optimizer_actor)
    graphdef_opt_c , state_opt_c = nnx.split(optimizer_critic)

    # Beginning training -- tracking the time as well
    print(f"TRAINING. {NUM_ROLLOUTS} ROLLOUTS, {N_STEPS} STEPS PER")
    t0 = time.time()
    final_carry , episode_rewards = train_all_episodes(state_actor , state_critic , state_opt_a , state_opt_c , rng=rng_train)
    episode_rewards.block_until_ready()
    time_taken = time.time() - t0

    # Evaluating!!!
    print(f"Time taken (JAX)! {time_taken:.2f}")
    print(f"First 10 Episode Rewards: {episode_rewards[:10]}")
    print(f"Last 10 Episode Rewards: {episode_rewards[-10:]}")