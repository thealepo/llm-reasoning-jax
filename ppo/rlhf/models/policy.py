import jax
import jax.numpy as jnp
from flax import nnx

from .transformer import Transformer , TransformerConfig

class PolicyModel(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.transformer = Transformer(config , rngs=rngs)
        self.ln_f = nnx.LayerNorm(config.HIDDEN_SIZE , rngs=rngs)
        self.linear_head = nnx.Linear(config.HIDDEN_SIZE , config.VOCAB_SIZE , rngs=rngs)

    def __call__(self , input_ids):
        x = self.transformer(input_ids)
        x = self.ln_f(x)
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
        rng , rng_sample = jax.random.split(rng)
        batch , prompt_len = prompt.shape  # [batch , seq_len]
        total_len = prompt_len + max_new_tokens  # seq_len + max_new_tokens

        # Preallocating full buffer (to make function jit-able)
        buffer = jnp.zeros((batch , total_len) , dtype=jnp.int32)
        buffer = buffer.at[: , :prompt_len].set(prompt)

        # Writing tokens in
        for i in range(max_new_tokens):
            logits = self(buffer)
            next_token = jax.random.categorical(rng_sample , logits[: , prompt_len + i - 1 , :] , axis=-1)
            buffer = buffer.at[: , prompt_len + i].set(next_token)

        return buffer

if __name__ == "__main__":
    config = TransformerConfig()
    policy = PolicyModel(config , rngs=nnx.Rngs(0))

    input_ids = jnp.ones((4,32) , dtype=jnp.int32)
    log_probs = policy.log_probs_of(input_ids)

    assert log_probs.shape == (4,32) , f"Expected (4 , 32) but got {log_probs.shape}"
    assert jnp.all(log_probs <= 0) , f"Log Probs must be negative"
    print("Yay!")

    # generate() method tests@
    rng = jax.random.PRNGKey(42)
    prompt = jnp.ones((4,32) , dtype=jnp.int32)
    output = policy.generate(prompt , rng=rng)
    print(output.shape) # (4 , 288)