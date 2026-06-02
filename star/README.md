# JAX Implementation of STaR: Self-Taught Reasoner

This is a clean JAX/Gemma implementation of [STaR: Bootstrapping Reasoning with Reasoning](https://arxiv.org/abs/2203.14465) (Zelikman et al., 2022), evaluated on the GSM8K grade-school math benchmark (WIP).

![Alt Text](https://github.com/user-attachments/assets/4e1ac422-f25f-4901-ae48-db066ef7120a)

## What is STaR?

STaR is a supervised finetuning training technique that utilizes a model's own thinking to further and advance its reasoning capabilities. The core algorithm consists of:

```
for each iteration:
  for each training question:
    prompt the model to reason through the answer
    if the model gets it right:
      keep the (question , rationale , answer) tuples
    if the model gets it wrong:
      rationalize -- show the model the correct answer and ask it to reverse-engineer the reasoning path
  finetune the model on all collected (questions , rationale , answer) tuples
```

After finetuning, the new, improved model becomes the new starting point in the next iteration.

### Rationalization

To me, the most important insight of this work is **rationalization**. So, rather than discarding examples that the model got wrong, you give the model a second chance by giving it the correct answer to a question as a hint, and ask it to retrace the steps and make the reasoning chain. This increases the number of usable training examples per iteration, which makes the model ultimately stronger and creating step-by-step problem solving.

---

## Usage of Implementation

```python
from model import load_model
from data import load_gsm8k
from main import main

model , tokenizer , params = load_model()
test_dataset = load_gsm8k('test')

accuracy = main(
  model=model,
  tokenizer=tokenizer,
  params=params,
  test_dataset=test_dataset,
  num_iterations=3,
  use_rationalization=True
)
print(f"Test accuracy: {accuracy:.2%}")
```

---

## Learn More

The following links are to my YouTube videos in which I deep-dive into the concept and implementation:

- Video 1: [How can we use an LLM's own Thinking to Train itself? The STaR Paper Dissected](https://www.youtube.com/watch?v=yygfKMu1DZI)
- Video 2: [Build your own STaR Training Loop!](https://youtu.be/b1_9p77hdgw)
