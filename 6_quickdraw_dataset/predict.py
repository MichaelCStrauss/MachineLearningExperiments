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
from dataset import labels, train, test

# %%
model_time = '2020-07-18T08:46:20.373203'

logger.info("Loading saved model...")
model = tf.keras.models.load_model('saved_models/' + model_time)
logger.info("Model loaded.")

# %%
def show_images(imgs, num_rows, num_cols, predictions=None, labels=None, scale=1.5, title=None):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    figure, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if predictions:
            title = predictions[i]
            if title == labels[i]:
                ax.set_title(title)
            else:
                t = ax.set_title(f'{title} ({labels[i]})')
                plt.setp(t, color='r')

    return axes

# %%
with tf.device("/GPU:0"):
    X, y = next(iter(test))
    logger.info("Making predictions...")
    overall_loss, overall_accuracy = model.evaluate(test)
    predicted = np.argmax(model.predict(X), axis=-1)
    logger.info("Displaying graphs...")
    images = [np.reshape(x, (28, 28)) for x in X]
    predictions = [labels[i] for i in predicted]
    l = [labels[i] for i in y]
    show_images(images, 16, 8, predictions, l)
    plt.savefig(f'./outputs/predictions-{model_time}.png')
