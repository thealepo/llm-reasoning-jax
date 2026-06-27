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
from einops import rearrange
from ppo.rlhf.rlhf import advantages
from ppo.rlhf.utils.loss import old_log_probs
from star.model import generate

BETA = 0.01
MAX_NEW_TOKENS = 32
MU = 4  # ppo_epochs equivalency
G = 8

def generate_group(policy , input_ids , prompt_len , rng):
    # ONE generation
    def gen_one(rng , _):
        rng , rng_gen = jax.random.split(rng)
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

def compute_log_probs(policy , outputs , prompt_len):
    # outputs are [batch , G , total_len]
    flattened = rearrange(outputs , 'b g t -> (b g) t')
    log_probs = policy.log_probs_of(flattened)
    log_probs = rearrange(log_probs , '(b g) t -> b g t' , g=G)
    return log_probs[: , : , prompt_len:]  # [batch , G , response_ken]

def grpo_loss(log_probs_rl , old_log_probs , log_probs_sft , advantages , epsilon=0.2):
    ratio = jnp.exp(log_probs_rl - old_log_probs)  # [batch , G , response len]
    At = rearrange(advantages , 'b g -> b g 1')

    clipped = jnp.clip(ratio , 1-epsilon , 1+epsilon)

    policy_loss = -jnp.mean(jnp.minimum(ratio * At , clipped * At))
    kl = jnp.mean(log_probs_rl - log_probs_sft)

    return policy_loss + BETA * kl

@nnx.jit
def train_step(policy , optimizer , outputs , old_log_probs , log_probs_sft , advantages , prompt_len):
    # Computing loss function
    def loss_fn(policy):
        log_probs_rl = compute_log_probs(policy , outputs , prompt_len)
        return grpo_loss(log_probs_rl , old_log_probs , log_probs_sft , advantages)

    loss_val , grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(policy , grads)
    return loss_val

# generate group -> collect rewards -> calculate advantages -> loss
def train_batch(policy , reward , reference , optimizer , input_ids , prompt_len , rng):
    # Splitting keys
    rng , rng_gen = jax.random.split(rng)

    # Generating G completions
    full_generations , responses = generate_group(policy , input_ids , prompt_len , rng_gen)  # [batch , G , total_len] | [batch , G , total_len - prompt_len]

    # Computing log probs
    old_log_probs = compute_log_probs(policy , full_generations , prompt_len) # NOTE: FIND A BETTER NAME FOR TS
    log_probs_sft = compute_log_probs(reference , full_generations , prompt_len)
    # both are [Batch , G , response_len]

    # Rewards from the RM
    flat_responses = rearrange(responses , 'b g t -> (b g) t')
    flat_rewards = reward(flat_responses)
    rewards = rearrange(flat_rewards , '(b g) -> b g' , g=G)
    advantages = compute_advantages(rewards)

    # splits
    graphdef_policy , state_policy = nnx.split(policy)
    state_optimizer = optimizer.init(state_policy)

    # MU rewards
    def scan_fn(carry , _):
        state_policy , state_optimizer = carry

        
