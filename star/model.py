import jax
import jax.numpy as jnp
from gemma import gm

CHECKPOINT = gm.ckpts.CheckpointPath.GEMMA2_2B_PT  # base model for STaR, no instruction-tuning

def load_model(checkpoint=CHECKPOINT):
    params = gm.ckpts.load_params(checkpoint)
    model = gm.nn.Gemma2_2B()
    tokenizer = gm.text.Gemma2Tokenizer()

    return model , tokenizer , params

def copy_params(params):
    return jax.tree.map(lambda x: jnp.array(x) , params)

def make_sampler(model , params , tokenizer):
    sampler = gm.text.Sampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        sampling=gm.text.TopkSampling(k=50 , temperature=0.7)
    )
    return sampler

def generate(sampler , prompt , max_new_tokens):
    return sampler.sample(prompt , max_new_tokens=max_new_tokens)