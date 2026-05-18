import imageio
import matplotlib.pyplot as pyplot
import numpy as numpy

def record_episode(model , key , path='cartpole.gif'):
    env = gym.make('CartPole-v1')
    state , _ = env.reset()
    frames = []
    done = False

    while not done:
        frames.append(env.render())
        key , subkey = jax.random.split(key)

        action , _ = model.sample(jnp.array(state) , subkey)
        state , _ , terminated , truncated , _ = env.step(int(action))

        done = terminated or truncated

    env.close()
    imageio.mimsave(path , frames , fps=30)