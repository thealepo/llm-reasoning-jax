- [Actor-Critic and PPO Algorithms Explained](https://www.youtube.com/watch?v=nQDlbxMsf8Y&)
- [PPO From Scratch](https://www.youtube.com/watch?v=bayeOpv4HEM&)
- [Finetuning Language Models from Human Preferences (2019), Learning to Summarize from Human Feedback (2020), InstructGPT (2022)](<link>)
- [RLHF PPO in JAX](<link>)

---

Proximal Policy Optimization (PPO) is an RL algorithm that originally built upon Actor-Critic and Trust Relative Policy Optimization (TRPO). 

Actor-Critic implemented a two-model architecture, with a Policy model playing the Actor by producing a probability distribution on an action space, given a state; and a Value model playing the Critic of that distribution by generating a value V(s) used to calculate the Advantage, which determines on average, how much better or worse an Actor's output was given the state. This Advantage is what is used in place of the return (used in REINFORCE). What is important here is that we don't wait until our environment is fully terminated to calculate the return, and use that number to determine whether an action was good or not. We use the Critic's approximation to determine the validity of an action, and thus this gives us a better representation of what actions led to favorable outcomes.

However, there are some issues that arise from this. Namely, 

# next, speak on the issues of A2C and TRPO
# explain PPO and the algorithm