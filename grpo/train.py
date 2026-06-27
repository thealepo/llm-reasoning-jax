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
import optax

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

def grpo_loss(log_probs_rl , old_log_probs , log_probs_sft , advantages , epsilon=0.2):
    ratio = jnp.exp(log_probs_rl - old_log_probs)  # [batch , G , response len]
    At = rearrange(advantages , 'b g -> b g 1')

    clipped = jnp.clip(ratio , 1-epsilon , 1+epsilon)

    policy_loss = -jnp.mean(jnp.minimum(ratio * At , clipped * At))
    kl = jnp.mean(log_probs_rl - log_probs_sft)

    return policy_loss + BETA * kl

def train_step(graphdefs , state_policy , state_optimizer , outputs , old_log_probs , log_probs_sft , advantages , prompt_len):
    graphdef_policy , optimizer = graphdefs

    def loss_fn(params):
        log_probs_rl = compute_log_probs(graphdef_policy , params , outputs , prompt_len)
        return grpo_loss(log_probs_rl , old_log_probs , log_probs_sft , advantages)

    loss_val , grads = jax.value_and_grad(loss_fn)(state_policy)
    updates , new_optimizer_state = optimizer.update(grads , state_optimizer)
    new_policy_state = optax.apply_updates(state_policy , updates)

    return new_policy_state , new_optimizer_state , loss_val

# generate group -> collect rewards -> calculate advantages -> loss
def train_epoch(graphdefs , state_policy , state_optimizer , reward_model , input_ids , prompt_len , rng):

    rng , rng_generate_group = jax.random.split(rng)
    full_generations , responses = generate_group(graphdefs , state_policy , input_ids , prompt_len , rng_generate_group)
    old_log_probs = compute_log_probs(graphdefs , state_policy , full_generations , prompt_len)

    advantages = compute_advantages(rewards)

    # A step of GRPO
    def body_fn(carry , _):
        state_policy , state_optimizer = carry
        new_state_policy , new_state_optimizer , loss = train_step(
            graphdefs , state_policy , state_optimizer , responses , old_log_probs , ... , advantages , prompt_len
        )

        return (new_state_policy , new_state_optimizer) , loss

