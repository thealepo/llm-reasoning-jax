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
    rewards = jnp.array([1.0])
    values = jnp.array([0.5])
    last_value = jnp.float32(0.0)
    mask = jnp.array([1.0])

    advantages , returns = compute_gae(rewards , values , last_value , mask)
    print(advantages)  # 0.5
    print(returns) # 1.0