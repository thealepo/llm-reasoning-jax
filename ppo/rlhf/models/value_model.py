import jax
import jax.numpy as jnp
from flax import nnx

from .transformer import Transformer , TransformerConfig

class ValueModel(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.transformer = Transformer(config , rngs=rngs)
        self.value_head = nnx.Linear(config.HIDDEN_SIZE , 1 , rngs=rngs)

    def __call__(self , input_ids):
        x = self.transformer(input_ids)  # [batch , seq_len , hidden_size]
        x = self.value_head(x) # [batch , seq_len , 1]
        return x.squeeze(-1)  # [batch , seq_len]... .squeeze() removes any dimension size 1


if __name__ == "__main__":
    config = TransformerConfig()
    value_model = ValueModel(config , rngs=nnx.Rngs(0))

    input_ids = jnp.ones((4,32) , dtype=jnp.int32)
    mask = jnp.ones((4,32) , dtype=jnp.int32)
    x = value_model(input_ids)

    assert x.shape == (4,32) , f"Expected (4,32) but got {x.shape}"
    print(f"Output shape: {x.shape}")