from data import load_gsm8k , normalize_answer , parse_answer_string
from model import load_model , make_sampler , generate
from prompts import standard_prompt , rationalization_prompt
from finetune import finetune

def main(model , tokenizer , params , num_iterations=3 , max_new_tokens=256 , use_rationalization=True , workdir='./star_output'):
    train_dataset = load_gsm8k('train')

    sampler = make_sampler(model , params , tokenizer)

    for i in range(num_iterations):
        examples = []

        # Rationale Generation
        for question , answer , _ in train_dataset:
            prompt = standard_prompt(question)
            output = generate(sampler , prompt , max_new_tokens=max_new_tokens)

            try:
                pred_reasoning , pred_answer = parse_answer_string(output)
            except (IndexError , ValueError):
                continue
            
            if pred_answer == answer:
                examples.append((question , answer , pred_reasoning))
            else:
                if use_rationalization:
                    prompt = rationalization_prompt(question , answer)
                    output = generate(sampler , prompt , max_new_tokens=max_new_tokens)

                    try:
                        rat_reasoning , _ = parse_answer_string(output)
                    except (IndexError , ValueError):
                        continue
                    
                    examples.append((question , answer , rat_reasoning))

        params = finetune(model , params , examples , tokenizer , workdir=f'{workdir}/iter_{i}')
        sampler = make_sampler(model , params , tokenizer)

    return params