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

# has a vmap taste to it.
import jax
import jax.numpy as jnp
from flax import nnx

BETA = 0.01
MAX_NEW_TOKENS = 32
MU = 4  # ppo_epochs equivalency
G = 8

def generate_group(graphdefs , state_policy , input_ids , prompt_len , rng):
    # Unpacking
    graphdef_policy = graphdefs[0]
    B = input_ids.shape[0]  # batch , prompt_len

    # RNG Stuff
    rng_keys = jax.random.split(rng , B*G).reshape(B , G , 2)

    #

    # Merging
    policy = nnx.merge(graphdef_policy , state_policy)

    # Generate g ys from the policy, given the input_ids.
    # return should be [batch , g , prompt_len+max_new_tokens]
    