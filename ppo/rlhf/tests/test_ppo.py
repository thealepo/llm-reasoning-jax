import jax
import jax.numpy as jnp
from flax import nnx
import optax

from ..models.transformer import TransformerConfig
from ..models.policy import PolicyModel
from ..models.value_model import ValueModel
from ..models.reward_model import RewardModel
from ..utils.loss import policy_loss_fn , value_loss_fn , compute_KL_penalty
from ..utils.gae import compute_gae
import rlhf as RLHF

# SET
BATCH , PROMPT_LEN , MAX_NEW_TOKENS = 2 , 8 , 16
SEQ_LEN = MAX_NEW_TOKENS

# Some functions set
def make_config():
    return TransformerConfig

def make_models(config):
    policy = PolicyModel(config , rngs=nnx.Rngs(0))
    value = ValueModel(config , rngs=nnx.Rngs(1))
    reward = RewardModel(config , rngs=nnx.Rngs(2))
    reference = PolicyModel(config , rngs=nnx.Rngs(0))

    return policy , value , reward , reference

def make_dummy_response(rng):
    tokens = jax.random.randint(rng , (BATCH,SEQ_LEN) , 1 , 256)
    mask_patch = jnp.array([[1]*SEQ_LEN , [1]*(SEQ_LEN-4) + [0]*4])
    return (tokens * mask_patch).astype(jnp.int32)

# TESTS
#1

#2

