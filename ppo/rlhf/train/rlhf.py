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
