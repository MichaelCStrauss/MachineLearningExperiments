import torch
import pickle

file = open("data/glove.6B.100d.txt")

dictionary = {}

for line in file:
    splits = line.split(" ")
    word = splits[0]
    weights = torch.tensor([float(x) for x in splits[1:]])
    dictionary[word] = weights

torch.save(dictionary, open("embeddings.pickle", "wb"))

