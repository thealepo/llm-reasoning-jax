# Direct Preference Optimization

<img width="1865" height="396" alt="Image" src="https://github.com/user-attachments/assets/e3e0b557-de72-4da9-905f-3ae6ee14e383" />

Direct Preference Optimization (DPO) reparameterizes the reward function from PPO, into a clean classification loss. Essentially, your own Policy *becomes* the reward model itself.

This collapsed loss is:

<img width="1898" height="172" alt="Image" src="https://github.com/user-attachments/assets/9bdbe050-2d80-418d-a041-7f6052766834" />

---

## Go more In-Depth

The following link is to the focused YouTube video that goes into the concept, math, and JAX implementation:

- [Direct Preference Optimization: DPO Math & JAX Implementation [Road to Reasoning #8]](https://youtu.be/gO4_C0aun9A)

