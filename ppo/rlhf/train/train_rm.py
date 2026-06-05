import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
from dataclasses import dataclass


# NOTE: MAKE A NEW FILE FOR ALL THE TRANSFORMER ARCHITECTURE...

# hyperparams for a language model
@dataclass(frozen=True , kw_only=True , slots=True)
class Config:
    VOCAB_SIZE = 256
    SEQ_LEN = 32
    HIDDEN_SIZE = 64
    MLP_HIDDEN_SIZE = 4 * 64
    N_HEADS = 4
    N_LAYERS = 2

class MultiHeadSelfAttention(nnx.Module):
    def __init__(self , config: Config , rngs: nnx.Rngs):
        self.n_heads = config.N_HEADS
        self.head_size = config.HIDDEN_SIZE // config.N_HEADS
        self.output_size = config.HIDDEN_SIZE
        self.hidden_size = config.HIDDEN_SIZE

        self.Wq = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)
        self.Wk = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)
        self.Wv = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)
        self.Wo = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)

    def __call__(self , x):
        q , k , v = self.Wq(x) , self.Wk(x) , self.Wv(x)

        def mha_rearrange(m):
            return rearrange(m , 'b n (h d) -> b h n d' , h=self.n_heads)
        q , k , v = map(mha_rearrange , (q,k,v))

        scale = self.head_size ** -0.5

        attn_weights = jnp.einsum('b h i d , b h j d -> b h i j' , q , k) * scale

        seq_len = x.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len,seq_len) , dtype=jnp.bool_))
        causal_mask = causal_mask[jnp.newaxis , jnp.newaxis , : , :]

        attn_weights = jnp.where(causal_mask , attn_weights , float('-inf'))
        attn_weights = jax.nn.softmax(attn_weights , axis=-1)
        out = jnp.einsum('b h i j , b h j d -> b h i d' , attn_weights , v)

        out = rearrange(out , 'b h n d -> b n (h d)')
        out = self.Wo(out)

        return out

class MultiLayerPerceptron(nnx.Module):
    def __init__(self , config: Config , rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.HIDDEN_SIZE , 4 * config.HIDDEN_SIZE , rngs=rngs)
        self.fc2 = nnx.Linear(config.HIDDEN_SIZE * 4 , config.HIDDEN_SIZE , rngs=rngs)

    def __call__(self , x):
        x = self.fc1(x)
        x = nnx.gelu(x)
        return self.fc2(x)

class TransformerLayer(nnx.Module):
    def __init__(self , config: Config , rngs: nnx.Rngs):
        self.attn = MultiHeadSelfAttention(config , rngs)
        self.mlp = MultiLayerPerceptron(config , rngs)
        self.ln1 = nnx.LayerNorm(config.HIDDEN_SIZE , rngs=rngs)
        self.ln2 = nnx.LayerNorm(config.HIDDEN_SIZE , rngs=rngs)

    def __call__(self , x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nnx.Module):
    def __init__(self , config: Config , rngs: nnx.Rngs):
        self.wte = nnx.Embed(config.VOCAB_SIZE , config.HIDDEN_SIZE , rngs=rngs)
        self.wpe = nnx.Embed(config.SEQ_LEN , config.HIDDEN_SIZE , rngs=rngs)
        self.layers = nnx.List([TransformerLayer(config , rngs) for _ in range(config.N_LAYERS)])
        self.ln_f = nnx.LayerNorm(config.HIDDEN_SIZE , rngs=rngs)
        self.final = nnx.Linear(config.HIDDEN_SIZE , 1 , rngs=rngs)  # FOR THE RM

    def __call__(self , input_ids , mask):
        batch_size , seq_len = input_ids.shape
        positions = jnp.arange(seq_len)

        x = self.wte(input_ids) + self.wpe(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)

        last_token_index = mask.sum(axis=-1).astype(jnp.int32) - 1
        print(last_token_index)
        x = x[jnp.arange(x.shape[0]) , last_token_index]
        return self.final(x).squeeze(-1)

#===============================================================================================

def bradley_terry_loss(y_winner , y_loser):
    return -jnp.log(jax.nn.sigmoid(y_winner - y_loser))







#================================================================================================
if __name__ == "__main__":
    y_winner = 6.74
    y_loser = -2.31
    print(f'Loss in Instance 1: {bradley_terry_loss(y_winner,y_loser)}')
    y_winner = 0.3
    y_loser = 8.3
    print(f'Loss in Instance 2: {bradley_terry_loss(y_winner,y_loser)}')

    # Transformer test
    rng = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(rng)
    config = Config()
    model = Transformer(config , rngs=rngs)
    input_ids = jnp.ones((4,32) , dtype=jnp.int32)
    mask = jnp.ones((4,32))
    scores = model(input_ids , mask)
    assert scores.shape == (4,) , f"Epxected (4,) but got {scores.shape}"

    def loss_fn(model):
        scores = model(input_ids , mask)
        return scores.mean()
    grads = nnx.grad(loss_fn)(model)
    jax.tree.map(lambda g: print(jnp.any(jnp.isnan(g))) , grads)

    mask = jnp.array([
        [1,1,1,1,0,0,0,0],
        [1,1,0,0,0,0,0,0],
    ])
    input_ids = jnp.ones((2,8) , dtype=jnp.int32)
    scores = model(input_ids,mask)
    assert scores.shape == (2,)