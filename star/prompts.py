SYSTEM_PROMPT = '''\
Given the question, think step-by-step for the answer.
'''

FEW_SHOT_EXAMPLES = '''\
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA: 
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10
'''

def prompt(question):
    prompt = f'Q:{question}\nA: '
    return SYSTEM_PROMPT + FEW_SHOT_EXAMPLES + prompt

def rationalization_prompt(question , answer):
    prompt = f'Q:{question}\nA:{answer}. Given the answer, generate proper reasoning to reach this answer.'
    return SYSTEM_PROMPT + FEW_SHOT_EXAMPLES + prompt