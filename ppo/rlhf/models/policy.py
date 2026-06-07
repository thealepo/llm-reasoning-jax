import jax
import jax.numpy as jnp
from flax import nnx

from .transformer import Transformer , TransformerConfig

class PolicyModel(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.transformer = Transformer(config , rngs=rngs)
        self.linear_head = nnx.Linear(config.HIDDEN_SIZE , config.VOCAB_SIZE , rngs=rngs)

    def __call__(self , input_ids):
        x = self.transformer(input_ids)
        x = self.linear_head(x)
        return x

    def log_probs_of(self , input_ids):
        hidden_state = self.transformer(input_ids)  # [batch , seq_len , hidden_size]
        logits = self.linear_head(hidden_state)  # [batch , seq_len , vocab_size]
        log_probs = jax.nn.log_softmax(logits , axis=-1)  # [batch , seq_len , vocab_size]

        token_log_probs = jnp.take_along_axis(
            log_probs , input_ids[... , jnp.newaxis] , axis=-1
        ).squeeze(-1)  # [batch , seq_len]

        return token_log_probs

    # NOTE: I have looked through a lot of JAX resources and a lot of LLM help
    # But unsure how else to deal with this.Shapes must be static.
    # If prompt is long and causes suitable responses exceeded the max lengths
    # Unsure how to deal with the issue as of now
    # Example: prompt_len + max_new_tokens >> max_seq_len
    # wtf to do.
    def generate(self , prompt , rng , max_new_tokens=256):
        batch , prompt_len = prompt.shape  # [batch , seq_len]
        total_len = prompt_len + max_new_tokens  # seq_len + max_new_tokens

        # Preallocating full buffer (to make function jit-able)
        buffer = jnp.zeros((batch , total_len) , dtype=jnp.int32)
        buffer = buffer.at[: , :prompt_len].set(prompt)

        # Writing tokens in
        for i in range(max_new_tokens):
            rng , rng_sample = jax.random.split(rng)
            
            filled = buffer[: , :prompt_len+i]
            logits = self(filled)
            next_token = jax.random.categorical(
                rng_sample , logits[: , -1 , :] , axis=1
            )
            buffer = buffer.at[: , prompt_len + i].set(next_token)

        return buffer

    # attempt at JIT-able autoregressiv egeneration
    def generate(self , prompt , rng , max_new_tokens=MAX_NEW_TOKENS):
        batch , prompt_len = prompt.shape
        total_len = prompt_len + max_new_tokens

        # Setting a vuffer of size [batch , total_len]
        buffer = jnp.zeros((batch,total_len) , dtype=jnp.int32)
        buffer = buffer.at[: , :prompt_len].set(prompt)  # first part is prompt, second part is zeros ment to be filled.

        # Functioning to check if we are over the max_new_tokens
        def cond_fn(carry):
            i , _ , _ = carry
            return i < max_new_tokens

        def body_fn(carry):
            i , buffer , rng = carry
            rng , rng_sample = jax.random.split(rng)

            logits = self(buffer) # [batch , total_len , vocab]
            next_token = jax.random.categorical(
                rng_sample , logits[: , prompt_len + i - 1 , :] , axis=1
            ) # [batch]
            buffer = buffer.at[: , prompt_len + i].set(next_token)
            return i+1 , buffer , rng

        _ , buffer , _ = jax.while_loop(
            cond_fn , body_fn , (jnp.int32(0) , buffer , rng)
        )
        return buffer

if __name__ == "__main__":
    config = TransformerConfig()
    policy = PolicyModel(config , rngs=nnx.Rngs(0))
    rng = jax.random.PRNGKey(42)

    BATCH , PROMPT_LEN , MAX_TOKENS = 2 , 4 , 16
    prompt = jax.random.randint(rng , (BATCH,PROMPT_LEN) , 1 , config.VOCAB_SIZE)

    output = policy.generate(prompt , rng=rng , max_new_tokens=MAX_TOKENS)

    assert output.shape == (BATCH , PROMPT_LEN + MAX_TOKENS) , f"Bad shape: {output.shape}"
    print("checkmark on shape")

    assert jnp.array_equal(output[: , :PROMPT_LEN] , prompt)
    print("pass prompt")

    