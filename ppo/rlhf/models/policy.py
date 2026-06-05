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

    # NOTE: ERROR HERE IN THE GENERATE FUNCTION. WITH JAX SHAPE SIZES. WORK ON IT SOON.
    def generate(self , prompt , rng , max_new_tokens=256):
        input_ids = prompt
        for _ in range(max_new_tokens):
            rng , rng_sample = jax.random.split(rng , 2)
            logits = self(prompt)
            next_token_logits = logits[: , -1 , :]
            next_token = jax.random.categorical(rng_sample , next_token_logits , axis=-1)
            input_ids = jnp.concatenate([input_ids , next_token] , axis=1)
        return input_ids

if __name__ == "__main__":
    config = TransformerConfig()
    policy = PolicyModel(config , rngs=nnx.Rngs(0))

    input_ids = jnp.ones((4,32) , dtype=jnp.int32)
    log_probs = policy.log_probs_of(input_ids)

    assert log_probs.shape == (4,32) , f"Expected (4 , 32) but got {log_probs.shape}"
    assert jnp.all(log_probs <= 0) , f"Log Probs must be negative"
    print("Yay!")

    #
    rng = jax.random.PRNGKey(42)
    prompt = jnp.ones((4,32) , dtype=jnp.int32)
    output = policy.generate(prompt , rng=rng)
    print(output.shape)