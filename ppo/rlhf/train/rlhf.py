# Rollout (data collection):
#   generate y from policy
#   compute policy log_probs and save as old
#   compute reference log_provs
#   compute values
#   compute reward
#   r_t is 0 for all tokens except final
#   compute the KL penalty per token
#   r_t -= beta * KL penalty per token
#   calculate GAE (advantages and returns)
#
# Training:
#    update policy and value weights
#    probably inputs: (y , old_log_probs , advantages , returns)
import jax
import jax.numpy as jnp
from flax import nnx

from ..models.transformer import TransformerConfig
from ..models.policy import PolicyModel
from ..models.value_model import ValueModel
from ..models.reward_model import RewardModel
from ..utils.loss import policy_loss_fn , value_loss_fn , compute_KL_penalty
from ..utils.gae import compute_gae

BETA = 0.95

def rollout(state_policy , state_value_model , state_reward_model , state_reference_model , input_ids , prompt_len , rng):
    # NOTE: rng, rng_policy

    # Merges of graphs and states
    policy = nnx.merge(graphdef_policy , state_policy)
    reward = nnx.merge(graphdef_reward , state_reward_model)
    value = nnx.merge(graphdef_value , state_value_model)
    reference = nnx.merge(graphdef_reference , state_reference_model)

    # Generate y from the policy
    # NOTE: for-loop for generation in policy.py (improve?)
    rng , rng_gen = jax.random.split(rng)
    y = policy.generate(input_ids , rng_gen , max_new_tokens=32)  # 32 for now  # (batch , prompt_len + 32)

    # Compute log_probs and save as old_log_probs
    old_log_probs = policy.log_probs_of(y)[: , prompt_len:] # (batch , 32)
    log_probs_sft = reference.log_probs_of(y)[: , prompt_len:] # (batch , 32)

    # Compute values (of generated tokens)
    values = value(y)[: , prompt_len:] # (batch , 32)

    # Compute reward, only on the real token
    mask = (y != 0).astype(jnp.int32)
    r = reward(input_ids , mask) # (batch,)

    # Per token reward (0 until last)
    r_t = jnp.zeros((y.shape[0] , 32)) # 9batch, 32)
    r_t = r_t.at[: , 1].add(r)

    # KL Penalty
    kl = compute_KL_penalty(old_log_probs , log_probs_sft , beta=BETA)  # NOTE: set beta
    r_t = r_t - kl

    # GAE
    next_values = jnp.zeros((y.shape[0],))
    gen_mask = jnp.ones((y.shape[0],32))

    compute_gae_batch = jax.vmap(compute_gae , in_axes=(0,0,0,0))
    advantages , returns = compute_gae_batch(r_t , values , next_values , gen_mask)

    return y , old_log_probs , advantages , returns


def train_step(state_policy , state_value_model , state_opt_p , state_opt_v , y , prompt_len , old_log_probs , advantages , returns):
    response = y[: , prompt_len:]
    mask = jnp.ones((y.shape[0] , y.shape[1] - prompt_len))

    # Merging splits
    policy = nnx.merge(graphdef_policy , state_policy)
    value = nnx.merge(graphdef_value , state_value_model)
    optimizer_policy = nnx.merge(graphdef_opt_p , state_opt_p)
    optimizer_value = nnx.merge(graphdef_opt_v , state_opt_v)

    # Trace-level loss function
    def policy_loss(params):
        p = nnx.merge(graphdef_policy , params)
        return policy_loss_fn(p , response , old_log_probs , advantages , '''mask''')
    def value_loss(params):
        v = nnx.merge(graphdef_value , params)
        return value_loss_fn(v , response , returns , '''mask''')

    # Gathering losses, grads, and updating weights
    p_loss_val , p_grads = jax.value_and_grad(policy_loss)(nnx.state(policy))
    v_loss_val , v_grads = jax.value_and_grad(value_loss)(nnx.state(value))
    optimizer_policy.update(policy , p_grads)
    optimizer_value.update(value , v_grads)

    return (nnx.state(policy) , nnx.state(value) , nnx.state(optimizer_policy) , nnx.state(optimizer_value) , p_loss_val , v_loss_val)

def train_epoch():
    pass
    # fori_loop prob

#def train():
#    pass

if __name__ == "__main__":
    config = TransformerConfig()
    rng = jax.random.PRNGKey(0)
    batch , prompt_len = 4 , 16

    policy = PolicyModel(config , rngs=nnx.Rngs(0))
    reference = PolicyModel(config , rngs=nnx.Rngs(1))
    value = ValueModel(config , rngs=nnx.Rngs(2))
    reward = RewardModel(config , rngs=nnx.Rngs(3))

    graphdef_policy , state_policy = nnx.split(policy)
    graphdef_reference , state_reference = nnx.split(reference)
    graphdef_value , state_value = nnx.split(value)
    graphdef_reward , state_reward = nnx.split(reward)

    input_ids = jnp.ones((batch , prompt_len) , dtype=jnp.int32)

    y , old_log_probs , advantages , returns = rollout(
        state_policy , state_value , state_reward , state_reference , input_ids , prompt_len , rng
    )

    assert y.shape == (batch , prompt_len + 32) ,  f"y: {y.shape}"
    assert old_log_probs.shape == (batch , 32) , f"old_log_probs: {old_log_probs.shape}"
    assert advantages.shape == (batch , 32), f"advantages: {advantages.shape}"
    assert returns.shape == (batch , 32), f"returns: {returns.shape}"

    # Sanity checks
    assert jnp.all(old_log_probs <= 0) , "log probs must be <= 0"
    assert not jnp.any(jnp.isnan(advantages)) , "nan in advantages"
    assert not jnp.any(jnp.isnan(returns)) , "nan in returns"

    print("rollout OK")