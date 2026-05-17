import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import gymnasium as gym
from model import PolicyModel

# Hyperparameters
NUM_EPISODES = 1_000
GAMMA = 0.99
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
ENVIRONMENT = 'CartPole-v1'

# Setting the JAX keys and Environment (CartPole-v1 for this Implementation)
key = jax.random.PRNGKey(42)
env = gym.make(ENVIRONMENT)
state , _ = env.reset()

# Initializing the custom policy model, and optimizer
model = PolicyModel(
    input_size=state.shape[0],
    hidden_size=HIDDEN_SIZE,
    output_size=env.action_space.n,
    rngs=nnx.Rngs(42)
)
optimizer = nnx.Optimizer(model , optax.adam(LEARNING_RATE) , wrt=nnx.Param)

def loss_fn(model , states , actions , returns):
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    logits = jax.vmap(model)(states)
    log_probs = jax.nn.log_softmax(logits)
    action_log_prob = log_probs[jnp.arange(actions.shape[0]) , actions]
    return -jnp.mean(action_log_prob * returns)

@nnx.jit
def train_step(model , optimizer , states , actions , returns):
    grads = nnx.grad(loss_fn)(model , states , actions , returns)
    optimizer.update(model , grads)

for episode in range(NUM_EPISODES):
    # Generate one episode
    episode_data = []
    state , _ = env.reset()
    done = False

    key, episode_key = jax.random.split(key)
    while not done:
        episode_key , subkey = jax.random.split(episode_key)

        action , log_prob = model.sample(jnp.array(state) , subkey)

        next_state , reward , terminated , truncated , info = env.step(int(action))

        done = terminated or truncated

        episode_data.append((state , action , reward))

        state = next_state

    # Calculate the return G
    returns = []
    G = 0
    for _ , _ , reward in reversed(episode_data):
        G = reward + GAMMA * G
        returns.append(G)
    returns = returns[::-1]

    states = jnp.stack([jnp.array(data[0]) for data in episode_data])
    actions = jnp.array([data[1] for data in episode_data])
    returns = jnp.array(returns)

    # Update weights of Policy model
    train_step(model , optimizer , states , actions , returns)