# Direct Preference Optimization

Direct Preference Optimization (DPO) reparameterizes the reward function from PPO, into a clean classification loss. Essentially, your own Policy *becomes* the reward model itself.

This collapsed loss is: `L_DPO(π_θ; π_ref) = -E_(x,y_w,y_l)~D [ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]`

---

## Go more In-Depth

The following link is to the focused YouTube video that goes into the concept, math, and JAX implementation:

- [Direct Preference Optimization: DPO Math & JAX Implementation [Road to Reasoning #8]](https://youtu.be/gO4_C0aun9A)

