# Inputs: initial policy, reward model, task prompts, hyperparams epsilon beta and mu

# sample a batch of prompts
# sample g outputs for each prompt
# compute rewards for each output
# compute the advantage per-token for each output (using the grae)

# for 0 -> mu: update policy by maximizing GRPO objective (???)
# update reward model through continuos training (???)

#====

# prompt 1 | prompt 2 | prompt 3 | ... | prompt BATCH
# output 1 output 2 output 3 ... output g | output 1 output 2 output 3 ... output g | ... etc.
# compute rewards per

# has a vmap or pmap taste to it.