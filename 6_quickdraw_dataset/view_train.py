import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import d2l.tensorflow as d2l
from tqdm import tqdm

x = np.load('x.npy')
y = np.load('y.npy')
labels = np.load('labels.npy')

dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x)).batch(256).cache(filename="cached.data")

# %%
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

iter_data = next(iter(dataset))
images = [np.reshape(iter_data[0][i], (28, 28)) for i in range(20)]
l = [labels[iter_data[1][i]] for i in range(20)]
show_images(images, 2, 10, titles=l)
plt.show()