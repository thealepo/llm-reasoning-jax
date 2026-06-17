# we sample two reponses given a prompt, from our reference SFT model
# optimize policy model to minimize the DPO_LOSS, for the given reference model, dataset, and KL Beta.
import jax
import jax.numpy as jnp
from flax import nnx

BETA = 0.95

def dpo_loss(policy , reference , x , y_winner , y_loser , ):
    # per token log provs per model
    def get_token_log_probs(model , y):
        full_seq = jnp.concatenate([x,y] , axis=-1)
        logits = model(full_seq)
        log_probs = jax.nn.log_softmax(logits , axis=-1)

        targets = full_seq[: , 1:]
        log_probs = log_probs[: , :-1 , :]

        token_log_probs = jnp.take_along_axis(
            log_probs , targets[... , jnp.newaxis] , axis=-1
        ).squeeze(-1)

        # mask stuff
        prompt_len = x.shape[1]
        completion_len = y.shape[1]
        mask = jnp.arange(token_log_probs.shape[1])
        mask = (mask >- prompt_len-1) & (mask < prompt_len-1 + completion_len)
        return (token_log_probs * mask).sum(axis=-1)


    policy_log_probs_winner = get_token_log_probs(policy , y_winner)
    policy_log_probs_loser = get_token_log_probs(policy , y_loser)
    reference_log_probs_winner = get_token_log_probs(reference , y_winner)
    reference_log_probs_loser = get_token_log_probs(reference , y_loser)

    # DPO ratios
    log_ratio_winner = policy_log_probs_winner - reference_log_probs_winner
    log_ratio_loser = policy_log_probs_loser - reference_log_probs_loser

    # Loss
    logits_dpo = BETA * (log_ratio_winner - log_ratio_loser)
    loss = -jnp.mean(jax.nn.log_sigmoid(logits_dpo))

    return loss