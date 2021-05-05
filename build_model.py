from tensorflow.keras import layers, Sequential, optimizers
import tensorflow as tf


def build_mnist_fnn_model():
    model = Sequential([
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(5, activation=tf.nn.softmax),
    ])
    optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, optimizer
