import numpy as np
import matplotlib.pyplot as plt
from read_data import *
import cv2 as cv
from PIL import Image

def rbg_to_grayscale(images):
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])


def show_data(x_test, num):
    for i in range(num):
        plt.imshow(x_test[i])
        plt.axis('off')
        plt.show()


def segment(x_train, name):

    x_train_copy = rbg_to_grayscale(x_train)
    x_flag = np.zeros(x_train_copy.shape)
    # threshold_list = np.zeros(x_train_copy.shape[0])

    if name == 'x_train':
        threshold_list = np.load('x_train_threshold.npy')
    elif name == 'x_valid':
        threshold_list = np.load('x_valid_threshold.npy')

    print(x_flag.shape)
    x_segment = np.zeros(x_train_copy.shape)
    assert isinstance(x_train_copy, np.ndarray)
    for i in range(x_train_copy.shape[0]):
        # # 求图片的平均灰度
        # image_mean = np.mean(x_train_copy[i])
        #
        # # 存储全局icv和阈值
        # global_icv = 0
        # global_threshold = 0
        #
        # # 求阈值
        # for threshold in range(1, 256):
        #     sum_po = 0
        #     num_po = 0
        #     sum_pb = 0
        #     num_pb = 0
        #     for j in range(x_train_copy.shape[1]):
        #         for k in range(x_train_copy.shape[2]):
        #             if x_train_copy[i][j][k] > threshold:
        #                 sum_po += x_train_copy[i][j][k]
        #                 num_po += 1
        #             else:
        #                 sum_pb += x_train_copy[i][j][k]
        #                 num_pb += 1
        #     icv = (num_po/(255*255))*(sum_po/(num_po+1)-image_mean)**2 + (num_pb/(255*255))*(sum_pb/(num_pb+1)-image_mean)**2
        #     if icv > global_icv:
        #         global_icv = icv
        #         global_threshold = threshold
        # # 将阈值加入到阈值列表中
        # threshold_list[i] = global_threshold
        # print(global_threshold)
        global_threshold = threshold_list[i]
        # 求小于阈值的坐标，并加入到flag数组中
        for j in range(x_train_copy.shape[1]):
            for k in range(x_train_copy.shape[2]):
                if x_train_copy[i][j][k] > global_threshold:
                    x_flag[i][j][k] = 0
                    # x_train[i][j][k] = 0
                else:
                    x_flag[i][j][k] = 1
        x_segment[i] = x_flag[i] * x_train_copy[i]

        # show_data(x_train, 1)
    file_name1 = name+'_flag'
    file_name2 = name+'_threshold'
    file_name3 = name + '_segment'
    np.save(file_name1, x_flag)
    np.save(file_name2, threshold_list)
    np.save(file_name3, x_segment)
    return x_flag


if __name__ == '__main__':
    # x_train, y_train, x_valid, y_valid, x_test = load_data()
    # segment(x_train, 'x_train')
    # segment(x_valid, 'x_valid')
    x_segment = np.load('x_valid_segment.npy')
    show_data(x_segment, 10)