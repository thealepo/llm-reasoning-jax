import re
from datasets import load_dataset
from typing import NamedTuple

class GSM8KExample(NamedTuple):
    question: str
    gold_answer: str
    gold_reasoning: str

def load_gsm8k(split):
    dataset = load_dataset('openai/gsm8k' , 'main' , split=split)
    
    examples = []
    for row in dataset:
        question = row['question'].strip()
        answer_text = row['answer'].strip()

        gold_reasoning , gold_answer = parse_answer_string(answer_text)
        
        examples.append(
            GSM8KExample(
                question=question,
                gold_answer=gold_answer,
                gold_reasoning=gold_reasoning
            )
        )

        return examples


def parse_answer_string(answer_text):
    parts = answer_text.split('####')

    reasoning , raw_answer = parts[0].strip() , parts[1].strip()
    answer = normalize_answer(raw_answer)
    return reasoning , answer

def normalize_answer(raw_answer):
    raw_answer = raw_answer.strip()
    raw_answer = raw_answer.replace(',' , '')
    raw_answer = raw_answer.replace('$' , '')
    raw_answer = raw_answer.strip()

    try:
        as_float = float(raw_answer)
        if as_float == int(as_float):
            return str(int(as_float))
        else:
            return str(as_float)
    except ValueError:
        return raw_answer


if __name__ == '__main__':
    train = load_gsm8k('train')
    test = load_gsm8k('test')

    print(len(train))
    print(len(test))

    ex = train[2]
    print(ex.question)
    print(ex.gold_reasoning)
    print(ex.gold_answer)