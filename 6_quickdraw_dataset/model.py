# %%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from loguru import logger
import datetime

# %%
class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=num_channels, kernel_size=3, padding="same", strides=strides
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Conv2D(
                    filters=num_channels, kernel_size=3, padding="same", strides=strides
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        if use_1x1conv:
            self.conv1x1 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides
            )
        else:
            self.conv1x1 = None

    def call(self, X):
        Y = self.model(X)
        if self.conv1x1 is not None:
            Y += self.conv1x1(X)
        else:
            Y += X
        return tf.keras.activations.relu(Y)


class ResnetBlock(tf.keras.Model):
    def __init__(self, num_channels, num_residuals, strides=1):
        super().__init__()
        self.model = tf.keras.models.Sequential()
        for i in range(num_residuals):
            if i == 0:
                self.model.add(Residual(num_channels, True))
            else:
                self.model.add(Residual(num_channels, False))

    def call(self, X):
        return self.model(X)


#%%
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        ResnetBlock(1024, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=len(labels)),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)