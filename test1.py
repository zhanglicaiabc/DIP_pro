from read_data import *
from test import *
import numpy as np
import cv2

if __name__ == '__main__':
    # x_train, y_train, x_valid, y_valid, x_test = load_data()
    x_train_mean = np.load('x_train_mean.npy')
    x_valid_mean = np.load('x_valid_mean.npy')
    x_train_co_matrix = np.load('x_train_co_matrix.npy')
    x_valid_co_matrix = np.load('x_valid_co_matrix.npy')

    x_train_mean = (x_train_mean / 255)*10
    x_valid_mean = (x_valid_mean / 255)*10
    co = np.load('co_matrix.npy')
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
    x_train_co_matrix = np.reshape(x_train_co_matrix,(500, -1))
    x_valid_co_matrix = np.reshape(x_valid_co_matrix,(200, -1))

    x_train = np.concatenate((x_train_mean, x_train_co_matrix), axis=1)
    x_valid = np.concatenate((x_valid_mean, x_valid_co_matrix), axis=1)
    print(x_train.shape, x_valid.shape)