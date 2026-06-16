# we sample two reponses given a prompt, from our reference SFT model
# optimize policy model to minimize the DPO_LOSS, for the given reference model, dataset, and KL Beta.

def dpo_loss(x , policy , reference ,y_winner , y_loser , ):
    # per token log provs per model
    def get_token_log_probs(model , y):
        logits = model(y)
        log_probs = jax.nn.log_softmax(logits , axis=-1)
        token_log_probs = jnp.take_along_axis(
            log_probs , y[... , jnp.newaxis] , axis=2
        ).squeeze(-1)
        return token_log_probs.sum(axis=-1)

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