import cv2
import math
from read_data import *
import numpy as np
# 定义最大灰度级数
gray_level = 255


def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    print("图像的高宽分别为：height,width", height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    print("max_gray_level:", max_gray_level)
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)
    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = int(srcdata[j][i])
            cols = int(srcdata[j + d_y][i + d_x])
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    # con:对比度反应了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大。
    # eng:熵（Entropy, ENT）度量了图像包含信息量的随机性，表现了图像的复杂程度。当共生矩阵中所有值均相等或者像素值表现出最大的随机性时，熵最大。
    # agm:角二阶矩（能量），图像灰度分布均匀程度和纹理粗细的度量。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
    # idm:反差分矩阵又称逆方差，反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大。
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm


def test(img):
    print(img.shape)
    try:
        img_shape = img.shape
    except:
        print('imread error')
        return

    # 这里如果用‘/’会报错TypeError: integer argument expected, got float
    # 其实主要的错误是因为 因为cv2.resize内的参数是要求为整数
    img = cv2.resize(img, (img_shape[1] // 2, img_shape[0] // 2), interpolation=cv2.INTER_CUBIC)

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0 = getGlcm(img, 1, 0)
    # glcm_1=getGlcm(src_gray, 0,1)
    # glcm_2=getGlcm(src_gray, 1,1)
    # glcm_3=getGlcm(src_gray, -1,1)

    asm, con, eng, idm = feature_computer(glcm_0)

    return [asm, con, eng, idm]


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test = load_data()
    x_train_copy = x_train.transpose(0, 3, 1, 2)
    x_valid_copy = x_valid.transpose(0, 3, 1, 2)
    # co = np.zeros((5, 3, 4))
    # for i in range(x_train_copy.shape[0]):
    #     for j in range(x_train_copy.shape[1]):
    #         result = test(x_train_copy[i][j])
    #         result = np.asarray(result, dtype=np.float32)
    #         co[y_train[i]][j] += result
    # co = co/100
    # print(co)
    # np.save('co_matrix', co)
    #
    co = np.load('co_matrix.npy')
    print(co)
    #
    # x_valid_co_matrix = np.load('x_valid_co_matrix.npy')
    # pred = np.zeros(x_valid_copy.shape[0], dtype=int)
    # for i in range(x_valid_co_matrix.shape[0]):
    #     sub = np.zeros((5, 3))
    #     for j in range(5):
    #         sub[j][0] = co[j][0][0] - x_valid_co_matrix[i][0][0]
    #         sub[j][1] = co[j][1][0] - x_valid_co_matrix[i][1][0]
    #         sub[j][2] = co[j][2][0] - x_valid_co_matrix[i][2][0]
    #     sub = np.linalg.norm(sub, ord=2, axis=1)
    #     list_mean_norm_1 = sub.tolist()
    #     min_index_1 = list_mean_norm_1.index(min(list_mean_norm_1))
    #
    #     if min_index_1 == 4 and abs(x_valid_co_matrix[i][1][0]-co[3][1][0]) > abs(x_valid_co_matrix[i][1][0]-co[4][1][0]):
    #         pred[i] = 4
    # print(pred)
    # print(y_valid)
    # x_train_co_matrix = []
    # for i in range(x_train_copy.shape[0]):
    #     valid_co_matrix = np.zeros((3, 4))
    #     for j in range(x_train_copy.shape[1]):
    #         result = test(x_train_copy[i][j])
    #         result = np.asarray(result, dtype=np.float32)
    #         valid_co_matrix[j] += result
    #     x_train_co_matrix.append(valid_co_matrix)
    # x_train_co_matrix = np.asarray(x_train_co_matrix, dtype=np.float32)
    # print(x_train_co_matrix.shape)
    # np.save('x_train_co_matrix', x_train_co_matrix)

        # mean_sub = co-valid_co_matrix
        # mean_sub = mean_sub.reshape(5, -1)
        # # print(mean_sub)
        # mean_norm = np.linalg.norm(mean_sub, ord=2, axis=1)
        # # print(mean_norm)
        # list_mean_norm = mean_norm.tolist()
        # min_index = list_mean_norm.index(min(list_mean_norm))
        # pred[i] = min_index
    # print(pred)
    # print(y_valid)
    # correct_sum = 0
    # for i in range(y_valid.shape[0]):
    #     if pred[i] == y_valid[i]:
    #         correct_sum += 1
    # print('correct', correct_sum/200)
