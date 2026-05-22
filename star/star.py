

for i in range(5):
    # Rationale Generation
    for question , answer , _ in train_dataset:
        prompt = standard_prompt(question)
        output = generate(sampler , prompt , max_new_tokens=max_new_tokens)

        pred_reasoning , raw_pred_answer = parse_answer_string(output)
        pred_answer = normalize_answer(raw_pred_answer)
        
        if pred_answer == answer:
            #TODO
        else:
            if use_rationalization:
                prompt = rationalization_prompt(question , answer)
                output = generate(sampler , prompt , max_new_tokens=max_new_tokens)
                #TODO

    new_dataset = examples_to_hf_dataset('''examples''')
    current_model = finetune('''pass''')

    

