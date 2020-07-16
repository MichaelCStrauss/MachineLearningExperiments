# %%
import tensorflow as tf

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.reshape(x_train, (-1, 28, 28, 1))
x_test = tf.reshape(x_test, (-1, 28, 28, 1))
# %%
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, padding="same", activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding="same", activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation="relu"),
        tf.keras.layers.Dense(84, activation="sigmoid"),
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
    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test)

# %%
