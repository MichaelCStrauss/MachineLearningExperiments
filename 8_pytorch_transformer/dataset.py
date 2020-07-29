# %%
from typing import List
import torch
import csv
import nltk
import collections
import numpy as np
from tqdm import tqdm
import os
import pickle
from loguru import logger

nltk.download("punkt")
logger.info("Checking for .pickle file of tokens...")

tokens = None
if os.path.exists("data/tokens.pickle"):
    logger.info("Pickle file exists, loading it")
    tokenized_sentences = pickle.load(open("data/tokens.pickle", "rb"))
else:
    logger.info("Pickle file does not exist, creating it")
    file = "./data/imdb_master.csv"

    reader = csv.reader(open(file, encoding="latin-1"))

    sentences = []

    logger.info("Loading sentences")
    for index, review in enumerate(tqdm(reader)):
        if index == 0:
            continue
        review_sentences: List[str] = nltk.sent_tokenize(review[2].lower())
        for i in range(len(review_sentences) // 3):
            sentences.append(
                review_sentences[3 * i].replace("<br /><br />", " ")
                + " "
                + review_sentences[3 * i + 1].replace("<br /><br />", " ")
                + " "
                + review_sentences[3 * i + 2].replace("<br /><br />", " ")
            )

    tokenized_sentences = []
    logger.info("Tokenising sentences")
    for sentence in tqdm(sentences):
        token_list: List[str] = nltk.word_tokenize(sentence)
        token_list.insert(0, "<sos>")
        token_list.append("<eos>")
        tokenized_sentences.append(token_list)
    pickle.dump(tokenized_sentences, open("data/tokens.pickle", "wb"))

# %%
tokens = [tk for sentence in tokenized_sentences for tk in sentence]
count = collections.Counter(tokens)

# %%
token_freqs = sorted(count.items(), key=lambda x: x[1], reverse=True)

min_freq = 3
unique_tokens = [token for token, freq in token_freqs if freq >= min_freq]

# %%
class TokenIndex:
    def __init__(self, unique_tokens):
        index_to_token = []
        token_to_index = {}
        for idx, token in enumerate(unique_tokens):
            index_to_token.append(token)
            token_to_index[token] = idx
        token_to_index["<unk>"] = len(index_to_token)
        index_to_token.append("<unk>")
        token_to_index["<pad>"] = len(index_to_token)
        index_to_token.append("<pad>")
        self.index_to_token = np.array(index_to_token)
        self.token_to_index = token_to_index

    def get_index(self, token):
        if token in self.token_to_index:
            return self.token_to_index[token]
        else:
            return self.token_to_index["<unk>"]

    def get_token(self, index):
        return self.index_to_token[index]


token_index = TokenIndex(unique_tokens)
vocab_size = len(token_index.index_to_token)
logger.info(
    f"Loaded {len(tokenized_sentences)} sentences, {vocab_size} unique tokens in vocab"
)

# %%
seq_length = 64
corpus = [
    [token_index.get_index(tk) for tk in sentence]
    for sentence in tokenized_sentences
    if len(sentence) <= seq_length and len(sentence) > 0
]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.corpus = data

    def __getitem__(self, idx):
        item = self.corpus[idx]
        mask = [1.0 for _ in range(len(item))]
        while len(item) <= seq_length:
            item.append(token_index.token_to_index["<pad>"])
            mask.append(0.0)

        return (
            torch.tensor(item[0:-1]).to("cuda"),
            torch.tensor(item[1:]).to("cuda"),
            torch.tensor(mask[0:-1]).to("cuda"),
        )

    def __len__(self):
        return len(self.corpus)


dataset = Dataset(corpus)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
