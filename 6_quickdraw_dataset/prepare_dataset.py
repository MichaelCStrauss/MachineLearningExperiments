# %%
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import d2l.tensorflow as d2l
from tqdm import tqdm
import gc

# %%
def load_data():
    """ Prepare train, validation and test datasets. """
    x = []
    y = []
    labels = []
    files = os.listdir('./data')
    for label, file in tqdm(enumerate(files)):
        if label > 50:
            break
        labels.append(os.path.splitext(os.path.basename(file))[0])
        file_path = os.path.join('./data', file)
        file_data = np.load(file_path)
        for index, image in enumerate(file_data):
            if index > 5_000:
                break
            reshaped = np.reshape(image.view(), (28, 28, 1))
            x.append(reshaped)
            y.append(label)
            del image
        del file_data
        gc.collect()
            
    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)
    return x, y, labels
x, y, labels = load_data()

np.save('x.npy', x)
np.save('y.npy', y)
np.save('labels.npy', labels)