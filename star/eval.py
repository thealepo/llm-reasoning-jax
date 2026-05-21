from data import load_gsm8k , normalize_answer
from model import load_model , make_sampler , generate
# make a prompts.py

def evaluate(model , params , tokenizer , test_examples , max_new_tokens=256):
    sampler = make_sampler(model , params , tokenizer)
    correct = 0

    for i , (question , answer , reasoning) in enumerate(test_examples):
        # prompt = some prompt-making func
        output = generate(sampler , prompt , max_new_tokens=max_new_tokens)
        # predicted = 

        if predicted == answer:
            correct += 1

    return correct / len(test_examples)