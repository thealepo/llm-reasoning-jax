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
from dpo.policy import prompt
import jax
import jax.numpy as jnp
from flax import nnx
from ppo.rlhf.rlhf import graphdef_policy
from einops import rearrange

BETA = 0.01
MAX_NEW_TOKENS = 32
MU = 4  # ppo_epochs equivalency
G = 8

def generate_group(graphdefs , state_policy , input_ids , prompt_len , rng):
    # Generate g ys from the policy, given the input_ids.
    # return should be [batch , g , prompt_len+max_new_tokens]
    graphdef_policy = graphdefs[0]

    def gen_one(rng , _):
        rng , rng_gen = jax.random.split(rng)
        policy = nnx.merge(graphdef_policy , state_policy)
        output = policy.generate(input_ids , rng=rng_gen , max_new_tokens=MAX_NEW_TOKENS)
        return rng , output  # output.shape is [batch , total_len]

    _ , outputs = jax.lax.scan(gen_one , rng , None , length=G)  # [G , batch , total_len]
    outputs = rearrange(outputs , 'g b t -> b g t') # [batch , G , total_len]
    responses = outputs[: , : , prompt_len:]
    return outputs , responses

def compute_advantages(rewards):
    # rewards.shape == [B , G]
    mean = rewards.mean(axis=1 , keepdims=True)
    std = rewards.std(axis=1 , keepdims=True) + 1e-8
    return (rewards - mean) / std  # [batch , G]

def compute_log_probs(graphdefs , state_policy , outputs , prompt_len):
    # outputs are [batch , G , total_len]
    graphdef_policy = graphdefs[0]
    policy = nnx.merge(graphdef_policy , state_policy)

    flattened = rearrange(outputs , 'b g t -> (b g) t')
    log_probs = policy.log_probs_of(flattened)
    log_probs = rearrange(log_probs , '(b g) t -> b g t' , g=G)
    return log_probs[: , : , prompt_len:]  # [batch , G , response_ken]
