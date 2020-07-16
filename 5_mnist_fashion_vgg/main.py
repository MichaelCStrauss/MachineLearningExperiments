# %%
import tensorflow as tf

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.reshape(x_train, (-1, 28, 28, 1))
x_test = tf.reshape(x_test, (-1, 28, 28, 1))

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(1000).batch(256)

# %%
def vgg_block(conv_layers, num_channels):
    block = tf.keras.Sequential()
    for _ in range(conv_layers):
        block.add(
            tf.keras.layers.Conv2D(
                num_channels, kernel_size=3, padding="same", activation="relu"
            )
        )
    block.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return block


# %%
model = tf.keras.models.Sequential(
    [
        vgg_block(3, 128),
        vgg_block(2, 256),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10),
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# %%
with tf.device("/GPU:0"):
    model.fit(dataset, epochs=3)

    model.evaluate(x_test, y_test)


# %%
