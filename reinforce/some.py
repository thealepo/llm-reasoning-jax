import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import gymnasium as gym
from model import PolicyModel

env = gym.make('CartPole-v1')
state , _ = env.reset()

model = PolicyModel(
    input_size=state.shape[0],
    hidden_size=8,
    output_size=env.action_space.n,
    rngs=nnx.Rngs(42)
)
optimizer = nnx.Optimizer(model , optax.adam(1e-3) , wrt=nnx.Param)

def loss_fn(model , episode_data , returns):
    log_probs = jnp.array([data[2] for data in episode_data])
    returns = jnp.array(returns)

    return -jnp.mean(log_probs * returns)

for episode in range(10_000):
    # Generate one episode
    episode_data = []
    state , _ = env.reset()
    done = False
    key = jax.random.PRNGKey(episode)

    while not done:
        key , subkey = jax.random.split(key)

        action , log_prob = model(jnp.array(state) , subkey)

        next_state , reward , terminated , truncated , info = env.step(int(action))

        done = terminated or truncated

        episode_data.append((state , action , log_prob , reward , next_state , done))

        state = next_state

    # Calculate the return G
    returns = []
    G = 0
    gamma = 0.99

    for _ , _ , _ , reward , _ , _ in reversed(episode_data):
        G = reward + gamma * G
        returns.append(G)
    returns = returns[::-1]

    # Update weights of Policy model
    grads = nnx.grad(loss_fn)(model , episode_data , returns)
    optimizer.update(model , grads)