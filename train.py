from tensorflow import keras
from tensorflow.keras import datasets
import tensorflow as tf
import numpy as np
from read_data import *
from test import *

def preprocess_mnist(x_in, y_in):
    # x_in = tf.cast(x_in, dtype=tf.float32) / 255
    # x_in = tf.reshape(x_in, [-1, 28 * 28])
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_in = tf.one_hot(y_in, depth=4)
    return x_in, y_in

# 执行自定义训练过程
def mnist_cap_fnn_train(model, optimizer):
    # 初始化模型
    model.build(input_shape=[500, 15])

    x_train, y_train, x_valid, y_valid, x_test = load_data()

    x_train_mean = np.load('x_train_mean.npy')
    x_valid_mean = np.load('x_valid_mean.npy')
    x_train_co_matrix = np.load('x_train_co_matrix.npy')
    x_valid_co_matrix = np.load('x_valid_co_matrix.npy')

    x_train_mean = (x_train_mean / 255) * 10
    x_valid_mean = (x_valid_mean / 255) * 10

    for i in range(x_train_co_matrix.shape[0]):
        for j in range(x_train_co_matrix.shape[1]):
            x_train_co_matrix[i][j][0] *= 100
            x_train_co_matrix[i][j][1] /= 100
            x_train_co_matrix[i][j][2] /= 10
    for i in range(x_valid_co_matrix.shape[0]):
        for j in range(x_valid_co_matrix.shape[1]):
            x_valid_co_matrix[i][j][0] *= 100
            x_valid_co_matrix[i][j][1] /= 100
            x_valid_co_matrix[i][j][2] /= 10
    x_train_co_matrix = np.reshape(x_train_co_matrix, (500, -1))
    x_valid_co_matrix = np.reshape(x_valid_co_matrix, (200, -1))

    x_train = np.concatenate((x_train_mean, x_train_co_matrix), axis=1)
    x_valid = np.concatenate((x_valid_mean, x_valid_co_matrix), axis=1)

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    valid_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_db = valid_db.shuffle(10000).map(preprocess_mnist).batch(128)

    # 执行训练过程
    for epoch in range(20):
        for step, (x_batch, y_batch) in enumerate(train_db):
            with tf.GradientTape() as tape:

                out = model(x_batch, training=True)
                out = tf.squeeze(out)

                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_batch, out, from_logits=False))
                loss_print = float(loss)

            # 执行梯度下降
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 获得对测试集的准确率
        acc = test(model, valid_db)

        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))

