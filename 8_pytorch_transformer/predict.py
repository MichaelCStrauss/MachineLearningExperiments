# %%
import click
import nltk
from model import build_model
from dataset import vocab_size, token_index
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda")


ntokens = vocab_size
model = build_model(device, ntokens)

checkpoint = torch.load("saved_models/model.pt")
model.load_state_dict(checkpoint["model_state"])
model.eval()

beam_depth = 2
max_length = 10
# %%
start_string = '<sos> as a fan of japanese film'
with torch.no_grad():
    input_tokens = start_string.split(" ")

    # def beam_search(input_tk, cumulative_probability=1, depth=0):
    #     if depth >= max_length:
    #         return [(input_tk, cumulative_probability)]

    #     # logger.info(
    #     #     f"Depth={depth}: input={input_tk}, prob={cumulative_probability}"
    #     # )
    #     predictions = F.softmax(model(input_indices))
    #     final_out: nn.Tensor = predictions[0][-1]

    #     values, indices = torch.topk(final_out, beam_depth + 1)

    #     outputs = []
    #     for i, index in enumerate(indices[0:-1]):
    #         if index == input_tk[-1]:
    #             index = indices[-1]
    #             i = len(indices) - 1
    #         prob = cumulative_probability * values[i]
    #         seq = torch.cat([input_tk, torch.tensor([index], device=device)])
    #         if token_index.get_token(index) == "<eos>":
    #             return [(seq, prob)]
    

    #         outputs.append(beam_search(seq, prob, depth + 1,))
    #     outputs = [item for sublist in outputs for item in sublist]
    #     return outputs

    context = torch.tensor(
        [token_index.get_index(tk) for tk in input_tokens], dtype=torch.long
    ).to(device).reshape((1, -1))

    def generate(context, ntok=20):
        for _ in range(ntok):
            out = model(context)
            logits = out[:, -1, :]
            indices_to_remove = logits < torch.topk(logits, 10)[0][..., -1, None]
            logits[indices_to_remove] = np.NINF
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            context = torch.cat([context, next_tok.unsqueeze(-1)], dim=-1)
        return context

    out = generate(context)
    print(
        [token_index.get_token(x) for x in out[0]]
    )


    # for i, prediction in enumerate(predictions):
    #     words = [token_index.get_token(int(torch.argmax(step))) for step in prediction]
    #     w = " ".join(words)
    #     logger.info(f"Prediction {i}: words={w}")




    # outputs = beam_search(input_indices)

    # outputs.sort(key=lambda x: x[1], reverse=True)
    # for indices, prob in outputs[0:5]:
    #     seq = " ".join([token_index.get_token(idx) for idx in indices])
    #     logger.info(f"P={prob}: {seq}")


