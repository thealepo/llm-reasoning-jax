import jax
import jax.numpy as jnp
from flax import nnx

EPSILON = 0.2

def policy_loss_fn(actor , response , old_log_probs , advantages , mask):
    # everything is (batch , seq_len)
    logits = actor(response) #(batch , seq_len , vocab_size)
    log_probs = jax.nn.log_softmax(logits , axis=-1) # (bathc , seq_len , vocab_size)

    # logits at position t predict token t+1, so token t is scored against position t-1
    token_log_probs = jnp.take_along_axis(
        log_probs[: , :-1 , :] , response[: , 1: , jnp.newaxis] , axis=2
    ).squeeze(-1) # (batch , seq_len - 1)

    # token 0 has no context, pad it to keep the output (batch , seq_len)
    pad = jnp.zeros((token_log_probs.shape[0] , 1) , dtype=token_log_probs.dtype)
    token_log_probs = jnp.concatenate([pad , token_log_probs] , axis=1) # (batch , seq_len)

    ratio = jnp.exp(token_log_probs - jax.lax.stop_gradient(old_log_probs))

    cpi = ratio * jax.lax.stop_gradient(advantages)
    clipped = jnp.clip(ratio , 1 - EPSILON , 1 + EPSILON) * jax.lax.stop_gradient(advantages)

    loss = -jnp.minimum(cpi , clipped)

    return jnp.sum(loss * mask) / jnp.sum(mask)

def value_loss_fn(critic , response , returns , mask):
    values = critic(response)
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