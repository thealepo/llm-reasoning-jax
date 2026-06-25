# Inputs: initial policy, reward model, task prompts, hyperparams epsilon beta and mu

# sample a batch of prompts
# sample g outputs for each prompt
# compute rewards for each output
# compute the advantage per-token for each output (using the grae)

# for 0 -> mu: update policy by maximizing GRPO objective (???)
# update reward model through continuos training (???)