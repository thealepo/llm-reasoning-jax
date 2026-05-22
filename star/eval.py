from data import load_gsm8k , normalize_answer
from model import load_model , make_sampler , generate
from prompts import standard_prompt

def evaluate(model , params , tokenizer , test_examples , max_new_tokens=256):
    sampler = make_sampler(model , params , tokenizer)
    correct = 0

    for i , (question , answer , reasoning) in enumerate(test_examples):
        prompt = standard_prompt(question)
        output = generate(sampler , prompt , max_new_tokens=max_new_tokens)
        
        raw_predicted = output.split('####')[-1].strip() if '####' in output else output.strip()
        predicted = normalize_answer(raw_predicted)

        if predicted == answer:
            correct += 1

    return correct / len(test_examples)

def run_ablations(model , tokenizer , base_params , no_rat_params , rat_params , test_examples):
    results = {}

    results['base'] = evaluate(model , base_params , tokenizer , test_examples)
    results['star_no_rationalization'] = evaluate(model , no_rat_params , tokenizer , test_examples)
    results['star_rationalization'] = evaluate(model , rat_params , tokenizer , test_examples)

    return results
