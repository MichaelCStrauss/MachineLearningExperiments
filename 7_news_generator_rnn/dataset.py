# %%
from typing import List
import csv
import itertools
import nltk

nltk.download("punkt")
import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import string
import os
import pickle
from loguru import logger

logger.info("Checking for .pickle file of tokens...")

tokens = None
if os.path.exists('data/tokens.pickle'):
    logger.info("Pickle file exists, loading it")
    tokens = pickle.load(open('data/tokens.pickle', 'rb'))
else:
    logger.info("Pickle file does not exist, creating it")
    file = "./data/Fake.csv"

    reader = csv.reader(open(file))

    sentences = []

    logger.info("Loading sentences")
    for index, article in enumerate(tqdm(reader)):
        if index == 0:
            continue
        if index > 10000:
            break
        article_sentences: List[str] = nltk.sent_tokenize(article[1].lower())
        for sentence in article_sentences:
            sentences.append(sentence)

    # %%
    tokens = []
    logger.info("Tokenising sentences")
    for sentence in tqdm(sentences):
        token_list: List[str] = nltk.word_tokenize(sentence)
        tokens.append('<start>')
        for token in token_list:
            if token.isalpha():
                tokens.append(token)
        tokens.append('<end>')
    pickle.dump(tokens, open('data/tokens.pickle', 'wb'))
logger.info(f'Loaded {len(tokens)} tokens')

# %%
count = collections.Counter(tokens)

# %%
token_freqs = sorted(count.items(), key=lambda x: x[1], reverse=True)

min_freq = 2
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

# %%
corpus = [token_index.get_index(tk) for tk in tokens]

# %%
seq_length = 20
dataset = tf.data.Dataset.from_tensor_slices(corpus).batch(
    seq_length + 1, drop_remainder=True
)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


batch_size = 64
dataset = (
    dataset.map(split_input_target).batch(batch_size, drop_remainder=True)
)
