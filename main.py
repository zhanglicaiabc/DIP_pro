from train import *
from build_model import *
from read_data import *
from segment import *
import numpy as np
from cifar10_cap_copy import *

if __name__ == '__main__':
    # 深度学习方法
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.85)
    # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    # session = tf.compat.v1.Session(config=config)
    conv_net, fc_net, optimizer2 = build_vgg13_model(0.0001)

    train_cifar10_copy(conv_net, fc_net, optimizer2)

    # x_train, y_train, x_valid, y_valid, x_test = load_data()
    # x_train_flag = np.load('x_train_flag.npy')
    # x_valid_flag = np.load('x_valid_flag.npy')

    # 算threshold均值
    # x_train_threshold = np.load('x_train_threshold.npy')
    # threshold_sum = np.zeros(5)
    # for i in range(x_train_threshold.shape[0]):
    #     threshold_sum[y_train[i]] += x_train_threshold[i]
    # print(threshold_sum / 100)

    # 求每个类的每个通道的总体均值
    # mean = np.zeros((5, 3))
    # num = np.zeros((5, 3))
    # for i in range(x_train.shape[0]):
    #     for j in range(x_train.shape[1]):
    #         for k in range(x_train.shape[2]):
    #             if x_train_flag[i][j][k] == 1:
    #                 mean[y_train[i]] += x_train[i][j][k]
    #                 num[y_train[i]] += 1
    # print(mean)
    # print(num)
    # mean = mean/num
    # np.save('mean', mean)
    # mean = np.load('mean.npy')
    # print(mean)
    # x_train_mean = []
    # pred = np.zeros(x_valid.shape[0], dtype=int)
    # x_valid_mean = np.load('x_valid_mean.npy')
    #
    # for i in range(x_train.shape[0]):
        # mean_valid = x_valid_mean[i]
    # for i in range(1):
    #     mean_valid = np.shape((1, 3))
    #     num_valid = 0
    #     for j in range(x_train.shape[1]):
    #         for k in range(x_train.shape[2]):
    #             if x_train_flag[i][j][k] == 1:
    #                 mean_valid += x_train[i][j][k]
    #                 num_valid += 1
    #     mean_valid = mean_valid/num_valid
    #     x_train_mean.append(mean_valid)
    # x_train_mean = np.asarray(x_train_mean, dtype=np.float32)
    # print(x_train_mean)
    # print(x_train_mean.shape)
    # np.save('x_train_mean', x_train_mean)
    #     mean_sub = mean-mean_valid
    #     mean_norm = np.linalg.norm(mean_sub, ord=2, axis=1)
    #     print(mean_norm)
    #     list_mean_norm = mean_norm.tolist()
    #     min_index = list_mean_norm.index(min(list_mean_norm))
    #     pred[i] = min_index
    # print(pred)
    # print(y_valid)
    #
    # correct_sum = 0
    # for i in range(y_valid.shape[0]):
    #     if pred[i] == y_valid[i]:
    #         correct_sum += 1
    # print('correct', correct_sum/200)