from data import load_gsm8k , normalize_answer , parse_answer_string
from model import load_model , make_sampler , generate
from prompts import standard_prompt , rationalization_prompt
from finetune import finetune


def main(num_iterations=3 , max_new_tokens=256 , use_rationalization=True , workdir='./star_output'):
    train_dataset , test_dataset = load_gsm8k('train') , load_gsm8k('test')

    model , tokenizer , params = load_model()
    sampler = make_sampler(model , params , tokenizer)

    for i in range(num_iterations):
        examples = []

        # Rationale Generation
        for question , answer , _ in train_dataset:
            prompt = standard_prompt(question)
            output = generate(sampler , prompt , max_new_tokens=max_new_tokens)

            pred_reasoning , raw_pred_answer = parse_answer_string(output)
            pred_answer = normalize_answer(raw_pred_answer)
            
            if pred_answer == answer:
                examples.append((question , answer , pred_reasoning))
            else:
                if use_rationalization:
                    prompt = rationalization_prompt(question , answer)
                    output = generate(sampler , prompt , max_new_tokens=max_new_tokens)

                    rat_reasoning , _ = parse_answer_string(output)
                    
                    examples.append((question , answer , rat_reasoning))

        params = finetune(model , params , examples , tokenizer , workdir=f'{workdir}/iter_{i}')
        sampler = make_sampler(model , params , tokenizer)

    return evaluate(model , params , tokenizer , test_dataset)
        

