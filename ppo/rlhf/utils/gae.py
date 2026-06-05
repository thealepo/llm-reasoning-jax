import jax
import jax.numpy as jnp
from flax import nnx

# Hyperparams
GAMMA = 0.99
LAMBDA = 0.95

def compute_gae(rewards , values , next_value , mask):
    next_values = jnp.append(values[1:] , next_value)
    deltas = rewards + GAMMA * next_values * mask - values

    def body_fn(last_advantage , delta):
        advantage = delta + GAMMA * LAMBDA * last_advantage
        return advantage , advantage

    _ , advantages = jax.lax.scan(
        body_fn,
        jnp.float32(0.0),
        deltas,
        reverse=True
    )

    returns = advantages + values
    return advantages , returns


# Unit testing
if __name__ == "__main__":
    # ONE
    rewards = jnp.array([1.0])
    values = jnp.array([0.5])
    last_value = jnp.float32(0.0)
    mask = jnp.array([1.0])

    advantages , returns = compute_gae(rewards , values , last_value , mask)
    print('ONE')
    print(advantages)  # 0.5
    print(returns) # 1.0
    print()

    # TWO
    rewards = jnp.array([1.0 , 0.6 , 1.3])
    values = jnp.array([0.8 , 1.9 , 0.1])
    last_value = jnp.float32(0.3)
    mask = jnp.array([1.0 , 1.0 , 1.0])

    advantages , returns = compute_gae(rewards , values , last_value , mask)
    print('TWO')
    print(advantages)
    print(returns)
    print()

    # THREE
    rewards = jnp.array([1.0 , 0.6 , 1.3])
    values = jnp.array([0.8 , 1.9 , 0.1])
    last_value = jnp.float32(0.3)
    mask = jnp.array([1.0 , 1.0 , 0.0])

    advantages , returns = compute_gae(rewards , values , last_value , mask)
    print('THREE')
    print(advantages)
    print(returns)
    print()