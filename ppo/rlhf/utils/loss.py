import jax
import jax.numpy as jnp
from flax import nnx

EPSILON = 0.2

def policy_loss_fn(actor , input_ids , old_log_probs , advantages , mask):
    # everything is (batch , seq_len)
    logits = jax.vmap(actor)(input_ids) #(batch , seq_len , vocab_size)
    log_probs = jax.nn.log_softmax(logits , axis=-1)

    ratio = jnp.exp(token_log_probs - jax.lax.stop_gradient(old_log_probs))

    cpi = ratio * jax.lax.stop_gradient(advantages)
    clipped = jnp.clip(ratio , 1 - EPSILON , 1 + EPSILON) * jax.lax.stop_gradient(advantages)

    loss = -jnp.minimum(cpi , clipped)

    return jnp.sum(loss * mask) / jnp.sum(mask)

def value_loss_fn(critic , input_ids , returns , mask):
    values = jax.vamp(critic)(input_ids)
    loss = (values - jax.lax.stop_gradient(returns)) ** 2
    return jnp.sum(loss * mask) / jnp.sum(mask)

# Unit Tests
if __name__ == "__main__":
    pass