# we sample two reponses given a prompt, from our reference SFT model
# optimize policy model to minimize the DPO_LOSS, for the given reference model, dataset, and KL Beta.

def dpo_loss(x , policy , reference ,y_winner , y_loser , ):
    # Log prob sampling
    logits_pw = policy(y_winner)
    log_probs_pw = jax.nn.log_softmax(logits_pw , axis=-1)
    token_log_probs_policy_winner = jnp.take_along_axis(
        log_probs_pw , y_winner[... , jnp.newaxis] , axis=2
    ).squeeze(-1)



    # Loss
    return -jnp.mean(jnp.log(jax.nn.sigmoid(BETA*jnp.exp(token_log_probs_policy_winner - token_log_probs_reference_winner)-BETA*(token_log_probs_policy_loser - token_log_probs_reference_loser))))