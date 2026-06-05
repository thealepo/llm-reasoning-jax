import jax
import jax.numpy as jnp

def bradley_terry_loss(y_winner , y_loser):
    return -jnp.log(jax.nn.sigmoid(y_winner - y_loser))

if __name__ == "__main__":
    y_winner = 6.74
    y_loser = -2.31
    print(f'Loss in Instance 1: {bradley_terry_loss(y_winner,y_loser)}')
    y_winner = 0.3
    y_loser = 8.3
    print(f'Loss in Instance 2: {bradley_terry_loss(y_winner,y_loser)}')