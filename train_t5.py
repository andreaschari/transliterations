import ir_datasets
import pandas as pd
import torch
import wandb
import logging
import argparse
import json

from pyterrier_t5 import mT5ReRanker
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
from tqdm import tqdm

BATCH_SIZE = 2
MAX_EPOCHS = 1
LEARNING_RATE = 5e-5 #5e-12 #5e-8 #1e-4 #5e-5
torch.cuda.empty_cache()
torch.manual_seed(0)

_logger = ir_datasets.log.easy()

OUTPUTS = ['yes', 'no']

argparser = argparse.ArgumentParser()
argparser.add_argument("--filepath", type=str, help="path to json file with triples", required=True)
argparser.add_argument("--output", type=str, help="path to save the model", required=True)

args = argparser.parse_args()
filepath = args.filepath
output = args.output

def iter_json_samples(filepath):
    # jsonl format is {"query": "text", "pos": List[str], "neg": List[str]}
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            yield 'Query: ' + data['query'] + ' Document: ' + data['pos'][0] + ' Relevant:', OUTPUTS[0]
            yield 'Query: ' + data['query'] + ' Document: ' + data['neg'][0] + ' Relevant:', OUTPUTS[1]


# train_iter = iter_train_samples()
train_iter = iter_json_samples(filepath=filepath)

model = MT5ForConditionalGeneration.from_pretrained("unicamp-dl/mt5-base-mmarco-v2").cuda()

# try this
for param in model.parameters():
  param.data = param.data.contiguous()

tokenizer = T5Tokenizer.from_pretrained("unicamp-dl/mt5-base-mmarco-v2")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

reranker = mT5ReRanker(verbose=False, batch_size=BATCH_SIZE)
reranker.REL = tokenizer.encode(OUTPUTS[0])[0]
reranker.NREL = tokenizer.encode(OUTPUTS[1])[0]

# Initialize wandb
wandb.init(project="mt5-training", config={
  "batch_size": BATCH_SIZE,
  "max_epochs": MAX_EPOCHS,
  "learning_rate": LEARNING_RATE,
  "model": "google/mt5-base",
  "training_data": filepath
})

epoch = 0
model.train()

_logger.info("Starting training")
_logger.info(f"Batch size: {BATCH_SIZE}")
_logger.info(f"Max epochs: {MAX_EPOCHS}")

while epoch < MAX_EPOCHS:
    total_loss = 0
    count = 0
    # 1996736 or 499184
    for _ in tqdm(range(1996736 // BATCH_SIZE), desc=f"Epoch {epoch}"):
      inp, out = [], []
      for _ in range(BATCH_SIZE):
        inp_, out_ = next(train_iter)
        inp.append(inp_)
        out.append(out_)
      inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.cuda()
      out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.cuda()
      loss = model(input_ids=inp_ids, labels=out_ids).loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      total_loss += loss.item()
      count += 1

      # Log loss to wandb
      wandb.log({"loss": total_loss / count})
    _logger.info(f'epoch {epoch} loss {total_loss / count}')
     # save the model
    model.save_pretrained(f'{output}/epoch-{epoch}')
    _logger.info("Saved model to disk")
    epoch += 1
# Log final checkpoint to wandb
# wandb.save(f'/root/nfs/CLIR/data/models/mt5-unicamp-RU_ZH_MMARCO_50_2M/epoch-final-{epoch}')
wandb.finish()
_logger.info("Finished training")