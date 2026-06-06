import jax
import jax.numpy as jnp
from flax import nnx

EPSILON = 0.2

def policy_loss_fn(actor , input_ids , old_log_probs , advantages , mask):
    # everything is (batch , seq_len)
    logits = actor(input_ids) #(batch , seq_len , vocab_size)
    log_probs = jax.nn.log_softmax(logits , axis=-1)

    token_log_probs = jnp.take_along_axis(
        log_probs , input_ids[... , jnp.newaxis] , axis=1
    ).squeeze(-1)

    ratio = jnp.exp(token_log_probs - jax.lax.stop_gradient(old_log_probs))

    cpi = ratio * jax.lax.stop_gradient(advantages)
    clipped = jnp.clip(ratio , 1 - EPSILON , 1 + EPSILON) * jax.lax.stop_gradient(advantages)

    loss = -jnp.minimum(cpi , clipped)

    return jnp.sum(loss * mask) / jnp.sum(mask)

def value_loss_fn(critic , input_ids , returns , mask):
    values = jax.vmap(critic)(input_ids)
    loss = (values - jax.lax.stop_gradient(returns)) ** 2
    return jnp.sum(loss * mask) / jnp.sum(mask)

def compute_KL_penalty(log_probs_rl , log_probs_sft , beta):
    return beta * (log_probs_rl - log_probs_sft)

# Unit Tests
if __name__ == "__main__":
    # Policy Loss
    # No ratio, just -mean(advantages)
    batch , seq_len = 2 , 4
    advantages = jnp.array([
        [1.0 , -0.5 , 0.6 , 0.8],
        [0.2 , 0.0 , -0.7 , 0.0]
    ])
    old_log_probs = jnp.zeros((batch , seq_len))
    mask = jnp.array([
        [1.0 , 1.0 , 1.0 , 1.0],
        [1.0 , 0.0 , 1.0 , 0.0]
    ])
    real_advantages = jnp.array([
        [1.0 , -0.5 , 0.6 , 0.8 , 0.2 , -0.7]
    ])
    expected = -jnp.mean(real_advantages)
    print(f'Expected Polucy Loss: {expected:.4}')

    # Value Loss
    values = jnp.array([
        [1.0 , 0.8 , 0.6 , 9.9],
        [0.5 , 0.3 , 9.4 , 9.4]
    ])
    returns = jnp.array([
        [1.0 , 0.8 , 0.6 , 0.0],
        [0.5 , 0.3 , 0.0 , 0.0]
    ])
    mask = jnp.array([
        [1.0 , 1.0 , 1.0 , 0.0],
        [1.0 , 1.0 , 0.0 , 0.0]
    ])
    loss = (values - returns) ** 2
    result = jnp.sum(loss * mask) / jnp.sum(mask)
    print(f'Expected Value loss: {result}')

    # KL Pentalty if same
    rl = jnp.array([0.34 , 0.87 , 0.43])
    sft = jnp.array([0.34 , 0.87 , 0.43])
    log_probs_rl , log_probs_sft = jnp.log(rl) , jnp.log(sft)
    KL_Penalty = compute_KL_penalty(log_probs_rl , log_probs_sft , beta=0.95)
    print(f'KL Penalthy: {KL_Penalty}')

    # KL Penalty if different
    rl = jnp.array([0.84 , 0.81 , 0.20])
    sft = jnp.array([0.34 , 0.87 , 0.43])
    log_probs_rl , log_probs_sft = jnp.log(rl) , jnp.log(sft)
    KL_Penalty = compute_KL_penalty(log_probs_rl , log_probs_sft , beta=0.95)
    print(f'KL Penalthy: {KL_Penalty}')