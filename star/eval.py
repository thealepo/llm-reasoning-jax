from data import load_gsm8k , normalize_answer
from model import load_model , make_sampler , generate
# make a prompts.py

def evaluate(model , params , tokenizer , test_examples , max_new_tokens=256):
    