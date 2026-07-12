# LLM Post-Training in JAX

<img width="3840" height="2160" alt="Image" src="https://github.com/user-attachments/assets/a7a1942c-236e-443d-bdb9-195f1ee070d9" />

Implementations of the core RL and preference-tuning algorithms behind LLM post-training, written from scratch in JAX (Flax NNX). Built alongside **Road to Reasoning**, a YouTube series walking through the math and the code for each one.

**[Watch the full playlist](https://youtube.com/playlist?list=PLCt2HxhgKZKyTjHLPPujPt4deV5vdQFno&si=b4Qo_4gLy96kYfSc)**

## Algorithms

| Algorithm | Description | Videos |
|---|---|---|
| [`reinforce/`](reinforce) | Vanilla policy gradient (REINFORCE), the starting point for everything else here. | [REINFORCE Explained](https://youtu.be/AtcpY_bdndY?si=gOsIxTXStBN0PXuz) |
| [`ppo/`](ppo) | Actor-Critic and PPO from scratch, plus a full RLHF pipeline (Policy, Value, Reward, and Reference models) with per-token KL penalty. | [Actor-Critic & PPO](https://www.youtube.com/watch?v=nQDlbxMsf8Y&) · [PPO from Scratch](https://www.youtube.com/watch?v=bayeOpv4HEM&) · [Core RLHF Papers](https://www.youtube.com/watch?v=Q0Of_6xJ0FQ&) · [RLHF Pipeline](https://youtu.be/iglZzhK1wrQ) |
| [`dpo/`](dpo) | Direct Preference Optimization, which reparameterizes the RLHF reward model into a closed-form classification loss, so the policy becomes its own reward model. | [DPO Math & JAX Implementation](https://youtu.be/gO4_C0aun9A) |
| [`grpo/`](grpo) | Group-Relative Policy Optimization drops the value model in favor of group-normalized advantages across sampled completions, with a clipped surrogate objective and KL penalty. | [DeepSeekMath](https://youtu.be/yBbmxXNoNUI?si=zoNwv_1otzfqlKan) · [GRPO from Scratch](https://youtu.be/cMgdk8R9MFc?si=fA27FvUgh7uIA2MT) |
| [`star/`](star) | STaR (Self-Taught Reasoner) bootstraps reasoning by finetuning a model on its own correct rationales, with rationalization for problems it initially gets wrong. Evaluated on GSM8K. | [The STaR Paper Dissected](https://www.youtube.com/watch?v=yygfKMu1DZI) · [Build your own STaR Loop](https://youtu.be/b1_9p77hdgw) |

Each subfolder has its own README with the algorithm's math, an implementation diagram, and links to the relevant videos.
