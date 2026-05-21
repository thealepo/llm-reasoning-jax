import os
import json
import optax
from gemma import gm
from kauldron import kd

def finetune(model , params , examples , tokenizer , workdir , num_steps=300 , batch_size=8 , lr=1e-3):
    os.makedirs(workdir , exist_ok=True)

    ds = kd.data.py.HuggingFace(
        dataset='''some data''',
        batch_size=batch_size,
        transforms=[
            gm.data.Seq2SeqTask(
                in_prompt='prompt',
                in_response='response',
                out_input='input',
                out_target='target',
                out_target_mask='loss_mask',
                tokenizer=tokenizer,
                max_length=256,
                truncate=True
            ),
        ],
    )