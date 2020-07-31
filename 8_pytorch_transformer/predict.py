# %%
import click
import nltk
from model import build_model
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer
from dataset import vocab_size, token_index

device = torch.device("cuda")

logger.info("Loading tokenizer")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# %%
logger.info("Building model...")
# ntokens = 50257
ntokens = vocab_size
model = build_model(device, ntokens)
model_dict = model.state_dict()

logger.info("Loading state...")
# trained_state = torch.load("saved_models/gpt2-pytorch_model.bin")
check_point = torch.load("saved_models/model.pt")
trained_state = check_point["model_state"]
pretrained_dict = {k: v for k, v in trained_state.items() if k in model_dict}
for key in trained_state.keys():
    if key not in model_dict:
        logger.warning(f"{key} not model")
for key in model_dict.keys():
    if key not in trained_state:
        logger.warning(f"{key} not in pretrained")
# print(model_dict.keys())
# print(trained_state.keys())

model_dict.update(pretrained_dict)

model.load_state_dict(model_dict)
model.out.weight = model.wte.weight

model.eval()

logger.info("State loaded")

# %%
start_string = "The actors"
beam_depth = 15
max_length = 2
# context = torch.tensor([tokenizer.encode(start_string)]).to(device)
context = torch.tensor(
    [[token_index.get_index(tk) for tk in start_string.lower().split(" ")]]
).to(device)


def decode(tokens):
    return " ".join([token_index.get_token(tk) for tk in tokens])


def beam_search(context, cumulative_probability=1, depth=0):
    if depth >= max_length:
        return [(context, cumulative_probability)]

    # logger.info(
    #     f"Depth={depth}: input={input_tk}, prob={cumulative_probability}"
    # )
    out = model(context)
    logits = out[:, -1, :]
    indices_to_remove = logits < torch.topk(logits, 20)[0][..., -1, None]
    logits[indices_to_remove] = np.NINF
    soft = F.softmax(logits, dim=-1)

    next_tok = torch.multinomial(soft, num_samples=beam_depth).squeeze(1)
    outputs = []
    for i, index in enumerate(next_tok[0]):
        prob = cumulative_probability * soft[0][index]
        next = index.reshape((1, 1))
        seq = torch.cat([context, next], dim=-1)

        outputs.append(beam_search(seq, prob, depth + 1,))

    outputs = [item for sublist in outputs for item in sublist]
    return outputs


num_iterations = 5
prob = 1
for _ in range(num_iterations):
    outputs = beam_search(context)
    outputs.sort(key=lambda x: x[1], reverse=True)
    context = outputs[0][0]
    logger.info(decode(context[0]))


for indices, prob in outputs[0:5]:
    seq = decode(indices[0])
    logger.info(f"P={prob}: {seq}")

