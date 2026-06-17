# pass in the data (x , y_winner , y_loser)
#   here there is still quesitons, but i presume concat x,ys and then feed the models the slice
# compute the loss
# update the policy & policy optimizer

def train_step():
    def loss_fn():
        return dpo_loss(policy , reference , y_winner , y_loser)

    pass

