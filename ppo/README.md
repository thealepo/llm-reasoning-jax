# **RLHF in JAX**

This is a Flax NNX implementation of Reinforcement Learning from Human Feedback (RLHF). This serves as a good learning resource to understanding the fundamentals of post-training, using PPO.

<img width="5424" height="5746" alt="Image" src="https://github.com/user-attachments/assets/b3fb86fe-490d-411e-82f3-851b4fdf2c74" />

---

Proximal Policy Optimization (PPO) is an RL algorithm that originally built upon Actor-Critic and Trust Relative Policy Optimization (TRPO). 

Actor-Critic implemented a two-model architecture, with a Policy model playing the Actor by producing a probability distribution on an action space, given a state; and a Value model playing the Critic of that distribution by generating a value V(s) used to calculate the Advantage, which determines on average, how much better or worse an Actor's output was given the state. This Advantage is what is used in place of the return (used in REINFORCE). What is important here is that we don't wait until our environment is fully terminated to calculate the return, and use that number to determine whether an action was good or not. We use the Critic's approximation to determine the validity of an action, and thus this gives us a better representation of what actions led to favorable outcomes.

However, there are some issues that arise from this. Namely, Actor-Critic has a limited ability to be scaled.

Introduce: PPO. PPO circumvents these problems in multiple ways, highlighted by the structure of the algorithm. Primarily, the implementation of a clipped surrogate allows their to be a limit in how much a policy gradient can update in a step. Additionally, PPO functions different,y from other policy gradient methods: we perform a rollout, in which we collect data from a smaller subset of the episode. Let's say, steps 1-128 of our full episode. And from there, we perform several updates to the policy.

---

## Reinforcement Learning from Human Feedback

RLHF for LLMs, while using PPO, uses four models which are simultaneously loaded into memory. Like vanilla PPO, we have two models that are trained online during the process: the Policy model, which generates a response y given a prompt x, and a Value model, which approximates the values for each token in the response that the Policy generated. The toher two models are frozen throughout this process: we have a Reward model, which functions as the environmental signal that outputs a scalar reward, given an input of the Policy's prompt and response given that prompt, and a Reference model, which is the original base Policy model (post-SFT), that we use to calculate a KL-penalty per-token, as to avoid excessive reward hacking and drifting from an original text-generating function from the Policy.

---

## Go more In-Depth

The following links are to the focused YouTube videos that go into the concepts and implementations:
- [Actor-Critic and PPO Algorithms Explained](https://www.youtube.com/watch?v=nQDlbxMsf8Y&)
- [PPO From Scratch](https://www.youtube.com/watch?v=bayeOpv4HEM&)
- [Finetuning Language Models from Human Preferences (2019), Learning to Summarize from Human Feedback (2020), InstructGPT (2022)](<link>)
- [RLHF PPO in JAX](<link>)
