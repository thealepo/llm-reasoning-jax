N = 50
# copy original model
current_model = model.copy()

for i in range(N):
    # rationale generation
    ra_pairs = []
    for question , answer in dataset:
        rationale , pred = current_model(question)
        correct = (pred == answer)
        ra_pairs.append((question , rationale , pred , answer , correct))
    
    # rationalization
    rationalized = []
    for i in range(len(ra_pairs)):
        question , rationale , pred , answer , correct = ra_pair[i]
        if not correct:
            new_rationale = model_n.rationalize(question , answer)
            rationalized.append((question , new_rationale , answer))
        else:
            rationalized.append((question , rationale , answer))

    # finetune original model on correct solutions
    current_model = tune(model , rationalized_ra_pairs)