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
from grpo.reward import RewardModel
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
import optax

BETA = 0.01
MAX_NEW_TOKENS = 8
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
    surrogate = jnp.minimum(ratio * At , clipped * At)

    # KL penalty (GRPO)
    kl = jnp.exp(log_probs_sft - log_probs_rl) - (log_probs_sft - log_probs_rl) - 1.0
    loss = -(surrogate - BETA * kl)

    return loss.mean()

@nnx.jit(static_argnames=('prompt_len',))
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
    flat_mask = jnp.ones_like(flat_responses)
    flat_rewards = reward(flat_responses , flat_mask)
    rewards = rearrange(flat_rewards , '(b g) -> b g' , g=G)
    advantages = compute_advantages(rewards)

    # MU rewards
    # NOTE: i wonder if there is any way I can make this whole function JIT-able. This is the bottleneck but not sure.
    losses = []
    for _ in range(MU):
        loss = train_step(policy , optimizer , full_generations , old_log_probs , log_probs_sft , advantages , prompt_len)
        losses.append(loss)

    return losses

if __name__ == "__main__":
    from grpo.policy import PolicyModel
    from grpo.reward import RewardModel
    from grpo.transformer import TransformerConfig

    # RNG
    rng = jax.random.PRNGKey(42)
    config = TransformerConfig()

    # hyo
    BATCH , PROMPT_LEN = 2 , 8

    # Model sand Optimizer
    policy = PolicyModel(config , rngs=nnx.Rngs(0))
    reference = PolicyModel(config , rngs=nnx.Rngs(0))
    reward = RewardModel(config , rngs=nnx.Rngs(1))
    optimizer = nnx.Optimizer(policy , optax.adam(1e-3) , wrt=nnx.Param)

    # Dummy data
    rng , rng_data = jax.random.split(rng)
    batches = [
        jax.random.randint(rng_data , (BATCH,PROMPT_LEN) , 1 , config.VOCAB_SIZE) for _ in range(5)
    ]

    for i , input_ids in enumerate(batches):
        rng , rng_batch = jax.random.split(rng)
        losses = train_batch(policy , reward , reference , optimizer , input_ids , PROMPT_LEN , rng_batch)
        mean_loss = sum(losses) / len(losses)

        loss_strs = [f'{float(l):.4f}' for l in losses]
        print(f'batch {i+1} | mean loss: {mean_loss:.4f} | losses: {loss_strs}')

    print('YAY')