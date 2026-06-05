import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
from dataclasses import dataclass

# Configuration for the Tranformer architecture to be used
@dataclass(frozen=True , kw_only=True , slots=True)
class TransformerConfig:
    VOCAB_SIZE = 256
    SEQ_LEN = 32
    HIDDEN_SIZE = 64
    MLP_HIDDEN_SIZE = 4 * 64
    N_HEADS = 4
    N_LAYERS = 2

# MHSA
class MultiHeadSelfAttention(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.n_heads = N_HEADS
        self.head_size = config.HIDDEN_SIZE // config.N_HEADS
        self.output_size = config.HIDDEN_SIZE
        self.hidden_size = config.HIDDEN_SIZE

        # Initializing the 4 Attention matrices (Query , Key, Value, Out)
        self.Wq = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)
        self.Wk = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)
        self.Wv = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)
        self.Wo = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)

    def __call__(self , x):
        # x shape is [batch , seq_len , hidden_size]
        # QKV Values
        Q , K , V = self.Wq(x) , self.Wk(x) , self.Wv(x)  # [batch , seq_len , hidden_size]

        # To account for the multiple heads, we rearrange the shape of our tensors
        def mha_rearrange(t):
            return rearrange(m , 'b n (h d) -> b h n d')  # [batch , n_heads , seq_len , head_dim]
        Q , K , V = map(mha_rearrange , (Q,K,V))

        # Scale 1 / sqrt(head_dim)
        scale = (self.head_size) ** -0.5

        # Computing self-attention
        attention_weights = (jnp.einsum('b n i d , b n h j -> b n i j') , Q , K) * scale  # QK^T / scale

        # Attention Causal map... to avoid tokens attending into future
        seq_len = x.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len,seq_len) , dtype=jnp.bool_)) # creates a lower triangular matrix size seq_len*seq_len
        causal_mask = causal_mask[jnp.newaxis , jnp.newaxis , : , :]

        attention_weights = jnp.where(causal_mask , attention_weights , float('-inf'))
        attention_weights = jax.nn.softmax(attention_weights , axis=-1)
        out = jnp.einsum('b n i j , b n j d -> b n i d')  # multiplying by V

        out = rearrange(out , 'b h n d -> b n (h d)') # Back to [batch , seq_len , hidden_size]
        out = self.Wo(out)

        return out

class MultiLayerPerceptron(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.HIDDEN_SIZE , config.MLP_HIDDEN_SIZE , rngs=rngs)
        self.fc2 = nnx.Linear(config.MLP_HIDDEN_SIZE , config.HIDDEN_SIZE , rngs=rngs)