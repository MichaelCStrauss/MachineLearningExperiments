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
from model import model
from dataset import test, train

model_time = datetime.datetime.utcnow().isoformat()

# %%
with tf.device("/GPU:0"):
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/" + model_time,
            save_weights_only=True,
            verbose=1,
            save_best_only=True,
        ),
    ]

    logger.info("Training")
    model.fit(
        train, epochs=10, validation_data=test, callbacks=callbacks,
    )

    model.save('saved_models/' + model_time)
