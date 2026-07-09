# Group-Relative Policy Optimization

<img width="5403" height="5102" alt="Image" src="https://github.com/user-attachments/assets/c0cfd448-ab8f-4519-bcec-5dab124cd7a0" />

Group-Relative Policy Optimization (GRPO) removes the value model from PPO entirely. So instead of learning a critic to estimate the advantage, GRPO samples a *group* of `G` completions per prompt and computes the advantage of each one relative to the group by normalizing each completion's reward against the group's mean and standard deviation.

```
A_i = (r_i - mean(r_1, ..., r_G)) / (std(r_1, ..., r_G) + eps)
```

This advantage is plugged into the same clipped surrogate objective as PPO, with an added per-token KL penalty (against a frozen reference policy):

```
KL = exp(log_pi_ref - log_pi_theta) - (log_pi_ref - log_pi_theta) - 1
```

Because the baseline comes from the group itself rather than a learned value function, GRPO needs one fewer model in memory and avoids the instability of training a critic alongside the policy.

---

## Go more In-Depth

The following links are to the focused YouTube videos that go into the concept, math, and JAX implementation:

- [DeepSeekMath: The Paper Behind GRPO](https://youtu.be/yBbmxXNoNUI?si=zoNwv_1otzfqlKan)
- [Let's Build GRPO From Scratch! (WIP)]()
