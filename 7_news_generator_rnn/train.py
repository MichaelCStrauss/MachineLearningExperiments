# %%
import numpy as np
import tensorflow as tf
from dataset import dataset, token_index, batch_size
from model import build_model
import hashlib

train = dataset.skip(65).shuffle(1000)
test = dataset.take(65)
model = build_model(batch_size)

summary = model.summary()
model_hash = hashlib.sha1(repr(summary).encode("ascii")).hexdigest()


# %%
with tf.device("/GPU:0"):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'saved_models/{model_hash}', save_weights_only=True
    )

    model.fit(train, epochs=2, callbacks=[checkpoint_callback], validation_data=test)
