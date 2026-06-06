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

def rollout(state_policy , state_value_model , state_reward_model , state_reference_model , input_ids , prompt_len , rng):
    # NOTE: rng, rng_policy

    # Merges of graphs and states
    policy = nnx.merge(graphdef_policy , state_policy)
    reward = nnx.merge(graphdef_reward , state_reward_model)
    value = nnx.merge(graphdef_value , state_value_model)
    reference = nnx.merge(graphdef_reference , state_reference_model)

    # Generate y from the policy
    # NOTE: for-loop for generation in policy.py (improve?)
    y = policy.generate(input_ids , '''rng_policy''' , max_new_tokens=32)  # 32 for now  # (batch , seq_len + new_tokens)

    # Compute log_probs and save as old_log_probs
    old_log_probs = policy.log_probs_of(y)[: , promp_len:]
    log_probs_sft = reference.log_probs_of(y)[: , prompt_len:]

    # Compute values
    values = value(y)

    # Compute reward
    r = reward(input_ids , mask)

    r_t = jnp.zeros(('''batch,seq_len''')) # with r at the end?

    # some reverse scan for KL penalty and GAE


def train_step():
    pass

def train_epoch():
    pass
    # fori_loop prob

#def train():
#    pass

if __name__ == "__main__":
    pass