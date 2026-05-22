import os
import json
import optax
from gemma import gm
from kauldron import kd

def examples_to_jsonl(examples, path):
    with open(path , 'w') as f:
        for question , answer , reasoning in examples:
            record = {
                'prompt': question,
                'response': f'{reasoning}\n#### {answer}'
            }
            f.write(json.dumps(record) + '\n')

def finetune(model , params , examples , tokenizer , workdir , num_steps=300 , batch_size=8 , lr=1e-3):
    os.makedirs(workdir , exist_ok=True)

    ckpt_dir = os.path.join(workdir , 'init_ckpt')
    gm.ckpts.save_params(params , ckpt_dir)

    data_dir = os.path.join(workdir , 'data')
    os.makedirs(data_dir , exist_ok=True)
    examples_to_jsonl(examples , os.path.join(data_dir, 'train.jsonl'))

    ds = kd.data.py.HuggingFace(
        path='json',
        split='train',
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=False,
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

    # Loss
    loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
        logits='preds.logits',
        labels='batch.target',
        mask='batch.loss_mask'
    )

    # Trainer!
    trainer = kd.train.Trainer(
        seed=42,
        workdir=os.path.join(workdir , 'trainer_out'),
        train_ds=ds,
        model=model,
        init_transform=gm.ckpts.LoadCheckpoint(path=ckpt_dir),
        num_train_steps=num_steps,
        train_loss={'loss': loss},
        optimizer=optax.adam(learning_rate=lr),
    )

    state , _ = trainer.train()

    return state.params