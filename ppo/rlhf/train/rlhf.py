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

BETA = 0.01
MAX_NEW_TOKENS = 32
PPO_EPOCHS = 4

# NOTE: MASK SHOULD LOWKEY JUST BE DEFINED OUTSIDE AND THEN PASSED THROUGH
def rollout(graphdefs , state_policy , state_value , state_reward , state_reference , input_ids , prompt_len , rng):
    # RNG stuff + graphdef unpacking
    rng , rng_gen = jax.random.split(rng)
    graphdef_policy , graphdef_value , graphdef_reward , graphdef_reference , _ , _ = graphdefs

    # Merge graphdefs and states
    policy = nnx.merge(graphdef_policy , state_policy)
    value = nnx.merge(graphdef_value , state_value)
    reward = nnx.merge(graphdef_reward , state_reward)
    reference = nnx.merge(graphdef_reference , state_reference)

    # Generate a y from the policy, given the input_ids
    y = policy.generate(input_ids , rng_gen , max_new_tokens=MAX_NEW_TOKENS)  # [batch , prompt_len + max_new_tokens]
    response = y[: , prompt_len:] # [batch , MAX_NEW_TOKENS]

    # Compute the log_probs for both policy and reference
    log_probs_rl = policy.log_probs_of(response)
    log_probs_sft = reference.log_probs_of(response)

    # Compute the reward (for the last token)
    mask = (response != 0).astype(jnp.float32)  # mask is 1 where token, 0 where no token

    # Values and Rewards
    r = reward(response , mask)
    values = value(response)

    # Make r_t
    last_index = mask.sum(axis=-1).astype(jnp.int32) - 1  # [batch]
    r_t = jnp.zeros_like(values) # [batch , seq_len]
    r_t = r_t.at[jnp.arange(r_t.shape[0]) , last_index].set(r)

    # KL Penalties
    kl_penalties = compute_KL_penalty(log_probs_rl , log_probs_sft , beta=BETA)
    r_t -= kl_penalties

    # Calculate the GAE
    next_value = jnp.zeros(values.shape[0])
    advantages , returns = jax.vmap(compute_gae)(r_t , values , next_value , mask)

    return y , response , log_probs_rl , advantages , returns , mask
    

def train_step(graphdefs , state_policy , state_value , state_opt_p , state_opt_v , response , old_log_probs , advantages , returns , mask):
    graphdef_policy , graphdef_value , _ , _ , graphdef_opt_p , graphdef_opt_v = graphdefs

    # Merging shit
    policy = nnx.merge(graphdef_policy , state_policy)
    value = nnx.merge(graphdef_value , state_value)
    optimizer_policy = nnx.merge(graphdef_opt_p , state_opt_p)
    optimizer_value = nnx.merge(graphdef_opt_v , state_opt_v)

    # Loss functions
    def policy_loss(params):
        p = nnx.merge(graphdef_policy , params)
        return policy_loss_fn(p , response , old_log_probs , advantages , mask)
    def value_loss(params):
        v = nnx.merge(graphdef_value , params)
        return value_loss_fn(v , response , returns , mask)

    policy_loss_val , policy_grads = jax.value_and_grad(policy_loss)(nnx.state(policy))
    value_loss_val , value_grads = jax.value_and_grad(value_loss)(nnx.state(value))
    optimizer_policy.update(policy , policy_grads)
    optimizer_value.update(value , value_grads)

    return (nnx.state(policy) , nnx.state(value) , nnx.state(optimizer_policy) , nnx.state(optimizer_value) , policy_loss_val , value_loss_val)

# NOTE: 1 rollout -> k epochs
def train_epoch(graphdefs ,state_policy , state_value , state_reward , state_reference , state_opt_p , state_opt_v , input_ids , prompt_len , rng):

    # Collect rollout
    rng , rng_rollout = jax.random.split(rng)
    y , response , old_log_probs , advantages , returns , mask = rollout(
        graphdefs , state_policy , state_value , state_reward , state_reference ,
        input_ids , prompt_len , rng_rollout
    )

    # A step of PPO
    def body_fn(carry , _):
        state_policy , state_value , state_opt_p , state_opt_v = carry
        state_policy , state_value , state_opt_p , state_opt_v , policy_loss_val , value_loss_val = train_step(
            graphdefs , state_policy , state_value , state_opt_p , state_opt_v , response , old_log_probs , advantages , returns , mask
        )
        return (state_policy , state_value , state_opt_p , state_opt_v) , (policy_loss_val , value_loss_val)

    # Initial carry _ loop
    init_carry = (state_policy , state_value , state_opt_p , state_opt_v)
    (state_policy , state_value , state_opt_p , state_opt_v) , (policy_losses , value_losses) = jax.lax.scan(
        body_fn , init_carry , None , length=PPO_EPOCHS
    )

    return (
        state_policy , state_value , state_opt_p , state_opt_v ,
        policy_losses , value_losses
    )

#def train():
#    pass

if __name__ == "__main__":
    pass