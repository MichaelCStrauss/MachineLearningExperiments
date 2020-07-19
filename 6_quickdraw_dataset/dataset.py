# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import d2l.tensorflow as d2l
from tqdm import tqdm
from loguru import logger
import datetime

x = (np.load("x.npy") / 255.0).astype(np.float32)
y = np.load("y.npy")
labels = np.load("labels.npy")
logger.info("Loaded training data")

dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x))
test = dataset.take(1001).batch(128)
train = dataset.skip(1001).batch(128)
logger.info("Created training and test dataset")