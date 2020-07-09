# %%
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt
import tensorflow as tf

# %%
true_w = tf.constant([3.0], shape=(1, 1))
true_b = -1
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
labels = labels + tf.random.normal(shape=labels.shape)
plt.plot(features, labels, "bo")

labels = tf.reshape(labels, (-1, 1))

# %%
def load_array(data, batch_size, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if is_train:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    return dataset


# %%
data = load_array((features, labels), 10, is_train=True)
initialiser = tf.initializers.RandomNormal(stddev=0.05)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initialiser))

loss = tf.keras.losses.MeanSquaredError()

trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

# %%
num_epochs = 5
for epoch in range(num_epochs):
    for X, y in iter(data):
        with tf.GradientTape() as t:
            l = loss(net(X, training=True), y)
        grads = t.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net.get_weights()[0]
print("error in estimating w", true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print("error in estimating b", true_b - b)

