# %%
import click
import nltk
from model import build_model
from dataset import vocab_size, token_index

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


@click.command()
@click.option("--model")
@click.option("--temperature", default=1.0)
@click.argument("start_string")
def predict(model, start_string, temperature):
    ntokens = vocab_size
    model = build_model(device, ntokens)

    checkpoint = torch.load("saved_models/model.pt")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    num_generate = 1
    num_words = 1
    with torch.no_grad():
        for sample in range(num_generate):

            input_tokens = start_string.split(' ')
            input_indices = torch.tensor(
                [token_index.get_index(tk) for tk in input_tokens], dtype=torch.long
            ).to(device)

            text_generated = input_tokens

            for i in range(num_words):
                predictions = F.softmax(model(input_indices))
                final_out: nn.Tensor = predictions[-1][-1]
                idx = final_out.argmax()
                print(token_index.get_token(idx))

            print(" ".join(text_generated))


if __name__ == "__main__":
    predict()
