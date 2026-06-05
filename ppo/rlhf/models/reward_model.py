import jax
import jax.numpy as jnp
from flax import nnx

from .transformer import Transformer , TransformerConfig

class RewardModel(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.transformer = Transformer(config , rngs=rngs)
        self.scalar_head = nnx.Linear(config.HIDDEN_SIZE , 1 , rngs=rngs)

    def __call__(self , input_ids , mask):
        x = self.transformer(input_ids)  # [batch , seq_len , hidden_size]

        # Extracting the last real token in a sequence
        last_token_index = mask.sum(axis=-1).astype(jnp.int32) - 1
        x = x[jnp.arange(x.shape[0]) , last_token_index]  # [batch , hidden_size]

        return self.scalar_head(x).squeeze(-1)  # [batch]

if __name__ == "__main__":
    config = TransformerConfig()
    reward_model = RewardModel(config , rngs=nnx.Rngs(0))

    input_ids = jnp.ones((4,32) , dtype=jnp.int32)
    mask = jnp.ones((4,32) , dtype=jnp.int32)
    x = reward_model(input_ids , mask)

    assert x.shape == (4,) , f"Expected (4,) but got {x.shape}"
    print(f"Output shape: {x.shape}")