# Rollout (data collection):
#   generate y from policy
#   compute policy log_probs and save as old
#   compute reference log_provs
#   compute values
#   compute reward
#   r_t is 0 for all tokens except final
#   compute the KL penalty per token
#   r_t -= beta * KL penalty per token
#   calculate GAE (advantages and returns)
#
# Training:
#    update policy and value weights
#    probably inputs: (y , old_log_probs , advantages , returns)
import time
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

BETA = 0.01
MAX_NEW_TOKENS = 32
PPO_EPOCHS = 4

# NOTE: MASK SHOULD LOWKEY JUST BE DEFINED OUTSIDE AND THEN PASSED THROUGH
def rollout(graphdefs , state_policy , state_value , state_reward , state_reference , input_ids , prompt_len , rng):
    # RNG stuff + graphdef unpacking
    rng , rng_gen = jax.random.split(rng)
    graphdef_policy , graphdef_value , graphdef_reward , graphdef_reference , _ , _ = graphdefs

    # Merge graphdefs and states
    policy = nnx.merge(graphdef_policy , state_policy)
    value = nnx.merge(graphdef_value , state_value)
    reward = nnx.merge(graphdef_reward , state_reward)
    reference = nnx.merge(graphdef_reference , state_reference)

    # Generate a y from the policy, given the input_ids
    y = policy.generate(input_ids , rng_gen , max_new_tokens=MAX_NEW_TOKENS)  # [batch , prompt_len + max_new_tokens]
    response = y[: , prompt_len:] # [batch , MAX_NEW_TOKENS]

    # Compute the log_probs for both policy and reference
    log_probs_rl = policy.log_probs_of(response)
    log_probs_sft = reference.log_probs_of(response)

    # Mask trick attempt (just no mask whatsoever)
    # fixed-length generation to resolve some complications I have having
    # no pad or eos necessary... for now
    mask = jnp.ones((y.shape[0] , MAX_NEW_TOKENS) , dtype=jnp.float32)

    # Values and Rewards
    r = reward(response , mask)  # NOTE: PASSING RESPONSE FOR NOW. IN THE FUTURE, THIS MUST HAVE ITS OWN MASK. SHAPE MISMATCH CAUSED THIS.
    values = value(response)

    # Make r_t
    last_index = mask.sum(axis=-1).astype(jnp.int32) - 1  # [batch]
    r_t = jnp.zeros_like(values) # [batch , seq_len]
    r_t = r_t.at[jnp.arange(r_t.shape[0]) , last_index].set(r)

    # KL Penalties
    kl_penalties = compute_KL_penalty(log_probs_rl , log_probs_sft , beta=BETA)
    r_t -= kl_penalties

    # Calculate the GAE
    next_value = jnp.zeros(values.shape[0])
    advantages , returns = jax.vmap(compute_gae)(r_t , values , next_value , mask)

    return y , response , log_probs_rl , advantages , returns , mask
    

def train_step(graphdefs , state_policy , state_value , state_opt_p , state_opt_v , response , old_log_probs , advantages , returns , mask):
    graphdef_policy , graphdef_value , _ , _ , opt_p , opt_v = graphdefs

    # Merging shit
    policy = nnx.merge(graphdef_policy , state_policy)
    value  = nnx.merge(graphdef_value ,  state_value)

    # Loss functions
    def policy_loss(params):
        p = nnx.merge(graphdef_policy , params)
        return policy_loss_fn(p , response , old_log_probs , advantages , mask)
    def value_loss(params):
        v = nnx.merge(graphdef_value , params)
        return value_loss_fn(v , response , returns , mask)

    # Gather losses and grads, update the model
    policy_loss_val , policy_grads = jax.value_and_grad(policy_loss)(nnx.state(policy))
    value_loss_val , value_grads = jax.value_and_grad(value_loss)(nnx.state(value))
    p_updates , new_opt_p = opt_p.update(policy_grads , state_opt_p)
    v_updates , new_opt_v = opt_v.update(value_grads , state_opt_v)
    new_state_policy = optax.apply_updates(nnx.state(policy), p_updates)
    new_state_value  = optax.apply_updates(nnx.state(value),  v_updates)

    return (new_state_policy , new_state_value , new_opt_p , new_opt_v , policy_loss_val , value_loss_val)

# NOTE: 1 rollout -> k epochs
def train_epoch(graphdefs ,state_policy , state_value , state_reward , state_reference , state_opt_p , state_opt_v , input_ids , prompt_len , rng):

    # Collect rollout
    rng , rng_rollout = jax.random.split(rng)
    y , response , old_log_probs , advantages , returns , mask = rollout(
        graphdefs , state_policy , state_value , state_reward , state_reference ,
        input_ids , prompt_len , rng_rollout
    )

    # A step of PPO
    def body_fn(carry , _):
        state_policy , state_value , state_opt_p , state_opt_v = carry
        state_policy , state_value , state_opt_p , state_opt_v , policy_loss_val , value_loss_val = train_step(
            graphdefs , state_policy , state_value , state_opt_p , state_opt_v , response , old_log_probs , advantages , returns , mask
        )
        return (state_policy , state_value , state_opt_p , state_opt_v) , (policy_loss_val , value_loss_val)

    # Initial carry _ loop
    init_carry = (state_policy , state_value , state_opt_p , state_opt_v)
    (state_policy , state_value , state_opt_p , state_opt_v) , (policy_losses , value_losses) = jax.lax.scan(
        body_fn , init_carry , None , length=PPO_EPOCHS
    )

    return (
        state_policy , state_value , state_opt_p , state_opt_v ,
        policy_losses , value_losses
    )
train_epoch = jax.jit(train_epoch , static_argnames=('graphdefs','prompt_len'))

def train(graphdefs , state_policy , state_value , state_reward , state_reference , state_opt_p , state_opt_v , data , prompt_len , rng , n_epochs=10):
    # Tracking losses
    policy_loss_history = []
    value_loss_history = []
    train_start = time.time()

    for epoch in range(n_epochs):
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_start = time.time()

        for batch_index , input_ids in enumerate(data):
            rng , rng_epoch = jax.random.split(rng)
            batch_start = time.time()

            state_policy , state_value , state_opt_p , state_opt_v , policy_losses , value_losses = train_epoch(
                graphdefs , state_policy , state_value , state_reward , state_reference , state_opt_p , state_opt_v , input_ids , prompt_len , rng_epoch
            )

            # Block before stopping clock
            policy_losses.block_until_ready()

            # Mena losses
            batch_time = time.time() - batch-start
            mean_p_loss = float(policy_losses.mean())
            mean_v_loss = float(value_losses.mean())
            epoch_policy_losses.append(mean_p_loss)
            epoch_value_losses.append(mean_v_loss)

            print(
                f"epoch [{epoch+1}/{n_epochs}] "
                f"batch [{batch_index+1}/{len(data)}] "
                f"policy_loss: {mean_p_loss:.4f} "
                f"value_loss: {mean_v_loss:.4f}"
                f"time: {batch_time:.2f}s"
            )
            if epoch == 0 and batch_index == 0:
                print(f"  (first batch includes JIT compile time)")

        epoch_time = time.time() - epoch_start
        epoch_mean_p = sum(epoch_policy_losses) / len(epoch_policy_losses)
        epoch_mean_v = sum(epoch_value_losses)  / len(epoch_value_losses)
        policy_loss_history.append(epoch_mean_p)
        value_loss_history.append(epoch_mean_v)

        print(f"--- epoch {epoch+1} summary | "
            f"mean policy_loss: {epoch_mean_p:.4f} | "
            f"mean value_loss: {epoch_mean_v:.4f} | "
            f"epoch_time: {epoch_time:.2f}s ---"
        )
        print()

    total_time = time.time() - train_start
    print(f"total training time: {total_time:.2f}s")

    return (state_policy , state_value , state_opt_p , state_opt_v , policy_loss_history , value_loss_history)

if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    config = TransformerConfig()

    BATCH , PROMPT_LEN , VOCAB_SIZE = 4 , 16 , config.VOCAB_SIZE

    # initialoizing models
    policy = PolicyModel(config , rngs=nnx.Rngs(0))
    value = ValueModel(config , rngs=nnx.Rngs(1))
    reward = RewardModel(config , rngs=nnx.Rngs(2))
    opt_p = optax.adam(1e-3)
    opt_v = optax.adam(1e-3)
    state_opt_p = opt_p.init(nnx.state(policy))
    state_opt_v = opt_v.init(nnx.state(value))

    # splits
    graphdef_policy , state_policy = nnx.split(policy)
    graphdef_value , state_value = nnx.split(value)
    graphdef_reward , state_reward = nnx.split(reward)
    graphdef_reference , state_reference = nnx.split(policy)

    graphdefs = (
        graphdef_policy , graphdef_value , graphdef_reward , graphdef_reference ,
        opt_p , opt_v
    )
    input_ids = jax.random.randint(rng , (BATCH,PROMPT_LEN) , 1 , VOCAB_SIZE)

    # THE TESTS
    print("TEST 1: rollout shapes")
    y , response , log_probs_rl , advantages , returns , mask = rollout(
        graphdefs , state_policy , state_value , state_reward , state_reference ,
        input_ids , PROMPT_LEN , rng
    )
    assert y.shape == (BATCH , PROMPT_LEN + MAX_NEW_TOKENS) , f"Bad: {y.shape}"
    assert response.shape == (BATCH , MAX_NEW_TOKENS) , f"Bad: {response.shape}"
    assert log_probs_rl.shape == (BATCH , MAX_NEW_TOKENS) , f'Bad: {log_probs_rl.shape}'
    assert advantages.shape == (BATCH , MAX_NEW_TOKENS) , f'Bad: {advantages.shape}'
    assert returns.shape == (BATCH , MAX_NEW_TOKENS) , f'Bad: {returns.shape}'
    assert mask.shape == (BATCH , MAX_NEW_TOKENS) , f'Bad: {mask.shape}'
    print("PASS TEST 1")
    print()

    print("TEST 2: prompt stayed the same")
    assert jnp.array_equal(y[: , :PROMPT_LEN] , input_ids) , "Bad!"
    print('PASS TEST 2')
    print()

    print('TEST 3: response length is correct')
    assert jnp.array_equal(response , y[: , PROMPT_LEN:]) , 'Bad!'
    print('PASS TEST 3')
    print()

    print('TEST 4: mask output (uhoh)')
    assert jnp.all(mask == 1.0) , f'Bad! got {mask}'
    print('PASS TEST 4')
    print()

    #===================================
    print('NAN STUFF')
    test = response
    hidden = policy.transformer(test)
    print('hidden nan:' , jnp.any(jnp.isnan(hidden)))
    print('hidden extremes:' , hidden.min() , hidden.max())

    logits = policy.linear_head(hidden)
    print('logit nan:' , jnp.any(jnp.isnan(logits)))
    print('logit extremes:' , logits.min() , logits.max())
    
    log_probs = jax.nn.log_softmax(logits , axis=-1)
    print('log prob nan:' , jnp.any(jnp.isnan(log_probs)))
    print('log prob extremes' , log_probs.min() , log_probs.max())

    # maybe attention
    x = policy.transformer.wte(test) + policy.transformer.wpe(jnp.arange(test.shape[1]))
    print("embedding nan:", jnp.any(jnp.isnan(x)))

    for i , layer in enumerate(policy.transformer.layers):
        attention_out = layer.mhsa(layer.ln1(x))
        print(f'layer {i} nan:' , jnp.any(jnp.isnan(attention_out)))
        mlp_out = layer.mlp(layer.ln2(x))
        print(f'layer {i} mlp nan' , jnp.any(jnp.isnan(mlp_out)))
        x += attention_out
        x += mlp_out
    print()
    #===================================

    print('TEST 5: log probs less than or eqla to zero')
    assert jnp.all(log_probs_rl <= 0) , f'Bad, got {log_probs_rl.max()}'
    print('PASS TEST 5')
    print()

    print('TEST 6: advantages and returns are finite')
    assert jnp.all(jnp.isfinite(advantages)) , f'NaN or Inf - advantafges'
    assert jnp.all(jnp.isfinite(returns)) , f'NaN or Inf - returns'
    print('PASS TEST 6')
    print()

    print('TEST 7: train_step losses')
    sp2 , sv2 , sop2 , sov2 , policy_loss_val , value_loss_val = train_step(
        graphdefs , state_policy , state_value , state_opt_p , state_opt_v ,
        response , log_probs_rl , advantages , returns , mask
    )
    assert jnp.isfinite(policy_loss_val) , f'not finite: {policy_loss_val}'
    assert jnp.isfinite(value_loss_val) , f'not finite: {value_loss_val}'
    print(f'PASS TEST 7: pol_loss:{policy_loss_val:.4f} and val_loss: {value_loss_val:.4f}')
    print()

    print('TEST 8: weights are ACTUALLY updated')
    graphdef_policy , *_ = graphdefs
    policy_test = nnx.merge(graphdef_policy , state_policy)
    def policy_loss_test(params):
        p = nnx.merge(graphdef_policy , params)
        return policy_loss_fn(p , response , log_probs_rl , advantages , mask)
    _ , grads = jax.value_and_grad(policy_loss_test)(nnx.state(policy_test))
    # finding grdeints
    before_leaves = jax.tree.leaves(state_policy)
    after_leaves = jax.tree.leaves(sp2)
    grad_leaves = jax.tree.leaves(grads)
    grad_norms = [jnp.abs(g).mean() for g in grad_leaves]
    biggest_index = int(jnp.argmax(jnp.array(grad_norms)))
    diffs = [jnp.abs(b - a).max() for b,a in zip(before_leaves , after_leaves)]
    max_diff_overall = max(diffs)
    biggest_diff_idx = int(jnp.argmax(jnp.array(diffs)))
    print("max diff across ALL leaves:", max_diff_overall)
    print("at index:" , biggest_diff_idx)
    print("before:" , before_leaves[biggest_diff_idx].mean())
    print("after:" , after_leaves[biggest_diff_idx].mean())
    assert max_diff_overall > 1e-7, f"Weights barely changed: {max_diff_overall}"
    print('PASS TEST 8')
    print()

    print('TEST 9: RNG STUF')
    y2 , response2 , log_probs2 , adv2 , ret2 , mask2 = rollout(
        graphdefs , state_policy , state_value , state_reward , state_reference ,
        input_ids , PROMPT_LEN , rng
    )
    assert jnp.array_equal(y , y2) , "Differs"
    assert jnp.array_equal(response , response2) ,  "Differes"
    assert jnp.allclose(advantages , adv2) , "Differs"
    print('PASS TEST 9')
    print()

    print("=======================================================================")
    print("ALL TESTS PAST")
    print("=======================================================================")
    print('\n\n\n\n')

    print('EPOCH TESTS')
    print('TEST 10: train_epoch runs and works, smoke test')
    sp3 , sv3 , sop3 , sov3 , policy_losses , value_losses = train_epoch(
        graphdefs , state_policy , state_value , state_reward , state_reference ,
        state_opt_p , state_opt_v , input_ids , PROMPT_LEN , rng
    )
    assert policy_losses.shape == (PPO_EPOCHS,) , f'Bad: {policy_losses.shape}'
    assert value_losses.shape == (PPO_EPOCHS,) , f'Bad: {value_losses.shape}'
    assert jnp.all(jnp.isfinite(policy_losses)) , f"Non-finite policy losses: {policy_losses}"
    assert jnp.all(jnp.isfinite(value_losses)) ,  f"Non-finite value losses: {value_losses}"
    print(f"policy losses across epochs: {policy_losses}")
    print(f"value losses across epochs:  {value_losses}")
    print('PASS TEST 10')
    print()

    print('TEST 11: any weight changes post epoch')
    diffs = [jnp.abs(b - a).max() for b,a in zip(jax.tree.leaves(state_policy) , jax.tree.leaves(sp3))]
    max_diff = max(diffs)
    assert max_diff > 1e-7 , f"Weights unchanged after train_epoch: {max_diff}"
    print(f"max weight diff after epoch: {max_diff:.6e}")
    print('PASS TEST 11')
    print()

    print('TEST 12: policy loss decrease across PPO epochs, within one actual epoch')
    first_loss = policy_losses[0]
    last_loss = policy_losses[-1]
    print(f"first PPO step loss: {first_loss:.4f}")
    print(f"last  PPO step loss: {last_loss:.4f}")
    assert last_loss < first_loss , f'Loss did not decrease'
    print('PASS TEST 12')
    print()

    print('TEST 13: optimizer state changed')
    opt_p_leaves_before = jax.tree.leaves(state_opt_p)
    opt_p_leaves_after  = jax.tree.leaves(sop3)
    opt_diffs = [jnp.abs(b - a).max() for b,a in zip(opt_p_leaves_before , opt_p_leaves_after)]
    assert max(opt_diffs) > 0 , "Adam moments never updated"
    print(f"max optimizer state diff: {max(opt_diffs):.6e}")
    print('PASS TEST 13')
    print('\n\n\n\n')


    print('============================================================================================================')
    print('.TRAIN() SMOKE TEST')
    print('============================================================================================================')

    # Dummy data
    N_BATCHES = 3
    rng , rng_data = jax.random.split(rng)
    data = [
        jax.random.randint(rng_data , (BATCH , PROMPT_LEN) , 1 , VOCAB_SIZE) for _ in range(N_BATCHES)
    ]

    state_policy , state_value , staet_opt_p , state_opt_v , policy_loss_history , value_loss_history = train(
        graphdefs , state_policy , state_value , state_reward , state_reference , state_opt_p , state_opt_v , data , PROMPT_LEN , rng , n_epochs=3
    )
    assert len(policy_loss_history) == 3 , "Should have one entry per epoch"
    assert len(value_loss_history)  == 3
    print("policy loss history:" , [f"{x:.4f}" for x in policy_loss_history])
    print("value loss history: " , [f"{x:.4f}" for x in value_loss_history])
    print("PASS: train() smoke test")