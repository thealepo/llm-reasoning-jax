# JAX Implementation of STaR: Self-Taught Reasoner

## What is STaR?

STaR is a supervised finetuning training technique that utilizes a model's own thinking to further and advance its reasoning capabilities. The core algorithm consists of '''do some pseudocode loop'''.

Rationalization is perhaps the most important aspect of this rather simple loop. This refers to the idea that instead of just disregarding the incorrect answers, we do an additional spet of letting the model see the answer to a specific question, and reverse-engineer how you go from the quesiton to the answer, thus resulting in a fit rationale despite initially getting the answer incorrect.

## Learn More

The following links are to my YouTube videos in which I deep-dive into the concept and implementation.
