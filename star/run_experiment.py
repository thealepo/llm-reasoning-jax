import os
import json
from data import load_gsm8k
from model import load_model
from eval import run_ablations
from star import main

if __name__ == '__main__':
    test_dataset = load_gsm8k('test')
    model , tokenizer , base_params = load_model()

    no_rat_params = main(model , tokenizer , base_params , use_rationalization=False , workdir='./output/no_rationalization')
    rat_params = main(model , tokenizer , base_params , use_rationalization=True , workdir='./output/rationalization')

    results = run_ablations(model , tokenizer , base_params , no_rat_params , rat_params , test_dataset)

    print(results)
    os.makedirs('./output' , exist_ok=True)
    with open('./output/results.json' , 'w') as f:
        json.dump(results , f , indent=2)