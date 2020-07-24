# %%
import numpy as np
import tensorflow as tf
from dataset import dataset, token_index
import hashlib

#%%
vocab_size = len(token_index.index_to_token)
embeddings = 128
rnn_units = 1024


def build_model(batch_size):
    # %%
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embeddings, batch_input_shape=[batch_size, None]
            ),
            tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            tf.keras.layers.Dense(vocab_size),
        ],
        name="GRU_Embeddings",
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model