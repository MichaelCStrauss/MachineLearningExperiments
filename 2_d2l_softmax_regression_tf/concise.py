# %%
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt
import tensorflow as tf

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

 # %%
initialiser = tf.initializers.RandomNormal(stddev=0.05)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
net.add(tf.keras.layers.Dense(10, kernel_initializer=initialiser))

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

trainer = tf.keras.optimizers.SGD(learning_rate=0.1)

# %%
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in train_iter:
        with tf.GradientTape() as t:
            l = loss(y, net(X))
        grads = t.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(y, net(X))
    print(f'epoch {epoch + 1}, loss {l:f}')

# %%
print("Final test accuracy:")
def accuracy(y_hat, y):  # @save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)
    return float((tf.cast(y_hat, dtype=y.dtype) == y).numpy().sum())


def evaluate_accuracy(net, data_iter):  # @save
    """Compute the accuracy for a model on a dataset."""
    sums = [0, 0]
    for _, (X, y) in enumerate(data_iter):
        sums[0] += accuracy(net(X), y)
        sums[1] += sum(y.shape)
    return sums[0] / sums[1]

print(evaluate_accuracy(net, test_iter))