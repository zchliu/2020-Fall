# 图像变换（图像、编程语言自选）
# 图像的参数化几何变换原理
# 图像的前向变换(forward warping)和逆向变换(inverse warping)
# 图像的下抽样原理和图像的内插方法原理（近邻插值和双线性插值）
# 完成图像的几何变换实验，包括：平移变换；旋转变换；欧式变换；相似变换；仿射变换；投影变换；
# 完成图像的高斯金字塔表示与拉普拉斯金字塔表示，讨论前置低通滤波和抽样频率的关系

# 特征检测
# 基于高斯一阶微分的图像梯度（幅值图和方向图），分析高斯方差对图像梯度的影响；
# 掌握canny边缘检测原理，完成图像的边缘检测实验，展示每个环节的处理结果（梯度图，NMS，边缘链接）；
# 掌握Harris焦点检测原理，完成图像的角点检测实验，分析窗口参数对角点检测的影响，讨论角点检测的不变性，等变性和定位精度；

import numpy as np
from cv2 import cv2 as cv
import math
from numpy import pi
import filter

epsilon = 1e-11


def geotransform(img, H, center):
    img_dim = len(img.shape)

    if img_dim == 3:
        row = img.shape[0]
        col = img.shape[1]
        channel = img.shape[2]

        result = np.zeros((row, col, channel))

        for i in range(row):
            for j in range(col):

                src_pos = np.dot(H, np.array([[i], [j], [1]]) - center) + center

                if (abs(src_pos[2, 0] - 1) > epsilon):

                    # 转换成齐次坐标
                    src_pos[0, 0] = src_pos[0, 0] / (src_pos[2, 0] + epsilon)
                    src_pos[1, 0] = src_pos[1, 0] / (src_pos[2, 0] + epsilon)
                    src_pos[2, 0] = 1

                src_pos = src_pos[0:2, 0:1]

                if 0 < src_pos[0, 0] < row - 1 and 0 < src_pos[1, 0] < col - 1:

                    left_up = np.array([[int(src_pos[0, 0])], [int(src_pos[1, 0])]])
                    right_up = np.array([[int(src_pos[0, 0])], [int(src_pos[1, 0]) + 1]])
                    left_down = np.array([[int(src_pos[0, 0]) + 1], [int(src_pos[1, 0])]])
                    right_down = np.array([[int(src_pos[0, 0]) + 1], [int(src_pos[1, 0]) + 1]])

                    for k in range(channel):

                        # 双线性插值
                        result[i, j, k] = img[left_up[0, 0], left_up[1, 0], k] * abs(right_down - src_pos)[0, 0] * abs(right_down - src_pos)[1, 0] + img[left_down[0, 0], left_down[1, 0], k] * abs(
                            right_up - src_pos)[0, 0] * abs(right_up - src_pos)[1, 0] + img[right_up[0, 0], right_up[1, 0], k] * abs(left_down - src_pos)[0, 0] * abs(
                                left_down - src_pos)[1, 0] + img[right_down[0, 0], right_down[1, 0], k] * abs(left_up - src_pos)[0, 0] * abs(left_up - src_pos)[1, 0]

        return result


def down_sample(img, kernel_size, sigma):

    img_dim = len(img.shape)

    if img_dim == 3:
        row = img.shape[0]
        col = img.shape[1]
        channel = img.shape[2]

        img_blur = filter.gauss_filter(img, kernel_size, sigma)
        dst_img = np.zeros((row // 2, col // 2, channel))
        for i in range(row // 2):
            for j in range(col // 2):

                src_pos = np.array([[2 * i], [2 * j]])
                for k in range(channel):

                    dst_img[i, j, k] = img_blur[src_pos[0, 0], src_pos[1, 0], k]

        dst_img = cv.convertScaleAbs(dst_img)
        return dst_img.astype(np.uint8)


def up_sample(img, kernel_size, sigma):

    img_dim = len(img.shape)

    if img_dim == 3:
        row = img.shape[0]
        col = img.shape[1]
        channel = img.shape[2]

        dst_img = np.zeros((2 * row, 2 * col, channel))
        for i in range(2 * row):
            for j in range(2 * col):
                if i % 2 == 0 and j % 2 == 0:
                    for k in range(channel):
                        dst_img[i, j, k] = img[i // 2, j // 2, k]

        ex_img = filter.padding(dst_img, (kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), "reflect")
        gauss_k = filter.gaussian_kernel(kernel_size, sigma)

        for i in range(2 * row):
            for j in range(2 * col):
                for k in range(channel):
                    if dst_img[i, j, k] == 0:
                        s = np.sum((ex_img[i:i + kernel_size, j:j + kernel_size, k] != 0) * gauss_k)
                        dst_img[i, j, k] = np.sum(ex_img[i:i + kernel_size, j:j + kernel_size, k] * gauss_k) / s

        dst_img = cv.convertScaleAbs(dst_img)
        return dst_img.astype(np.uint8)


def gaussian_pyramid(img, level, kernel_size, sigma):

    result = []
    dst_img = img.copy()
    dst_sigma = sigma
    result.append(dst_img)
    for i in range(level):

        dst_img = down_sample(dst_img, kernel_size, dst_sigma)
        dst_sigma = dst_sigma / 2
        result.append(dst_img)

    return result


def laplace_pyramid(img, level, kernel_size, sigma):

    result = []
    current_img = img.copy()
    for i in range(level):
        down_img = down_sample(current_img, 1, 1)
        up_img = up_sample(down_img, kernel_size, sigma)
        detail_img = current_img - up_img
        result.append(detail_img)
        current_img = down_img

    result.append(current_img)
    return result


def build_from_laplace_pyramid(laplace_pyramid, kernel_size, sigma):

    length = len(laplace_pyramid)
    current_img = laplace_pyramid[length - 1]

    for i in range(length - 2, -1, -1):
        up_img = up_sample(current_img, kernel_size, sigma)
        current_img = up_img + laplace_pyramid[i]

    return current_img.astype(np.uint8)


def DeriveOG_filter(img, kernel_size, sigma):

    # img_x为x方向上的梯度
    # img_y为y方向上的梯度
    kernel_x = np.zeros((kernel_size, kernel_size))
    kernel_y = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            norm = (i - kernel_size // 2)**2 + (j - kernel_size // 2)**2
            kernel_x[i, j] = (j - kernel_size // 2) * np.exp(-norm / (sigma**2 * 2))
            kernel_y[i, j] = (i - kernel_size // 2) * np.exp(-norm / (sigma**2 * 2))

    s_x = np.sum(abs(kernel_x)) / 2
    s_y = np.sum(abs(kernel_y)) / 2
    kernel_x = kernel_x / s_x
    kernel_y = kernel_y / s_y

    img_x = filter.conv(img, kernel_x, "copy")
    img_y = filter.conv(img, kernel_y, "copy")

    img_x = cv.convertScaleAbs(img_x)
    img_y = cv.convertScaleAbs(img_y)
    img_edge = 0.5 * img_x + 0.5 * img_y

    img_edge = (img_edge - np.min(img_edge)) / (np.max(img_edge) - np.min(img_edge)) * 255
    return img_edge


def NMS(img_edge):

    kernel_x = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    kernel_y = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    img_x = filter.conv(img_edge, kernel_x, "copy")
    img_y = filter.conv(img_edge, kernel_y, "copy")
    ex_img = filter.padding(img_edge, (1, 1), (1, 1), "copy")
    dst_img = np.zeros((img_edge.shape[0], img_edge.shape[1]))

    for i in range(img_edge.shape[0]):
        for j in range(img_edge.shape[1]):

            if np.sqrt(img_x[i, j]**2 + img_y[i, j]**2) > epsilon:
                gradiant = np.array([[0.5 * img_x[i, j] / np.sqrt(img_x[i, j]**2 + img_y[i, j]**2)], [0.5 * img_y[i, j] / np.sqrt(img_x[i, j]**2 + img_y[i, j]**2)]])
                forward_pos = np.array([[i - gradiant[1, 0]], [j - gradiant[0, 0]]])
                # gradiant[0] 为左右方向，gradiant[1]为上下方向

                left_up_f = np.array([[int(forward_pos[0, 0])], [int(forward_pos[1, 0])]])
                right_up_f = np.array([[int(forward_pos[0, 0])], [int(forward_pos[1, 0] + 1)]])
                left_down_f = np.array([[int(forward_pos[0, 0]) + 1], [int(forward_pos[1, 0])]])
                right_down_f = np.array([[int(forward_pos[0, 0]) + 1], [int(forward_pos[1, 0] + 1)]])

                forward_I = ex_img[left_up_f[0, 0], left_up_f[1, 0]] * abs(right_down_f - forward_pos)[0, 0] * abs(
                    right_down_f - forward_pos)[1, 0] + ex_img[left_down_f[0, 0], left_down_f[1, 0]] * abs(right_up_f - forward_pos)[0, 0] * abs(
                        right_up_f - forward_pos)[1, 0] + ex_img[right_up_f[0, 0], right_up_f[1, 0]] * abs(left_down_f - forward_pos)[0, 0] * abs(
                            left_down_f - forward_pos)[1, 0] + ex_img[right_down_f[0, 0], right_down_f[1, 0]] * abs(left_up_f - forward_pos)[0, 0] * abs(left_up_f - forward_pos)[1, 0]

                backward_pos = np.array([[i + gradiant[1, 0]], [j + gradiant[0, 0]]])
                left_up_b = np.array([[int(backward_pos[0, 0])], [int(backward_pos[1, 0])]])
                right_up_b = np.array([[int(backward_pos[0, 0])], [int(backward_pos[1, 0]) + 1]])
                left_down_b = np.array([[int(backward_pos[0, 0]) + 1], [int(backward_pos[1, 0])]])
                right_down_b = np.array([[int(backward_pos[0, 0]) + 1], [int(backward_pos[1, 0]) + 1]])

                backward_I = ex_img[left_up_b[0, 0], left_up_b[1, 0]] * abs(right_down_b - backward_pos)[0, 0] * abs(
                    right_down_b - backward_pos)[1, 0] + ex_img[left_down_b[0, 0], left_down_b[1, 0]] * abs(right_up_b - backward_pos)[0, 0] * abs(
                        right_up_b - backward_pos)[1, 0] + ex_img[right_up_b[0, 0], right_up_b[1, 0]] * abs(left_down_b - backward_pos)[0, 0] * abs(
                            left_down_b - backward_pos)[1, 0] + ex_img[right_down_b[0, 0], right_down_b[1, 0]] * abs(left_up_b - backward_pos)[0, 0] * abs(left_up_b - backward_pos)[1, 0]

                if img_edge[i, j] > backward_I and img_edge[i, j] > forward_I:
                    dst_img[i, j] = img_edge[i, j]
            elif np.sqrt(img_x[i, j]**2 + img_y[i, j]**2) < epsilon and img_edge[i, j] > epsilon:
                dst_img[i, j] = img_edge[i, j]

    return dst_img


def Hysteresis_thresholding(img, threshold1, threshold2):

    row = img.shape[0]
    col = img.shape[1]

    dst_img = np.zeros((row, col))
    seen = np.zeros((row, col))

    for i in range(row):
        for j in range(col):

            if img[i, j] > threshold1:
                dst_img[i, j] = 255

    new_add = 100000
    while new_add > 5:
        s = 0
        for i in range(row):
            for j in range(col):
                if dst_img[i, j] == 255:
                    seen[i, j] = 1
                    for k in range(i - 1, i + 2):
                        for l in range(j - 1, j + 2):
                            if 0 <= k < row and 0 <= l < col and seen[k, l] != 1:
                                if img[k, l] > threshold2:
                                    dst_img[k, l] = 255
                                    seen[k, l] = 1
                                    s += 1
        new_add = s

    return dst_img


def Harris_corner(img, kernel_size1, sigma1, kernel_size2, sigma2, k):

    # img_x为x方向上的梯度
    # img_y为y方向上的梯度
    kernel_x = np.zeros((kernel_size1, kernel_size1))
    kernel_y = np.zeros((kernel_size1, kernel_size1))
    for i in range(kernel_size1):
        for j in range(kernel_size1):
            norm = (i - kernel_size1 // 2)**2 + (j - kernel_size1 // 2)**2
            kernel_x[i, j] = (j - kernel_size1 // 2) * np.exp(-norm / (sigma1**2 * 2))
            kernel_y[i, j] = (i - kernel_size1 // 2) * np.exp(-norm / (sigma1**2 * 2))

    s_x = np.sum(abs(kernel_x)) / 2
    s_y = np.sum(abs(kernel_y)) / 2
    kernel_x = kernel_x / s_x
    kernel_y = kernel_y / s_y

    Ix = filter.conv(img, kernel_x, "copy")
    Iy = filter.conv(img, kernel_y, "copy")

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    Sx2 = filter.gauss_filter(Ix2, kernel_size2, sigma2)
    Sy2 = filter.gauss_filter(Iy2, kernel_size2, sigma2)
    Sxy = filter.gauss_filter(Ixy, kernel_size2, sigma2)

    row = Sx2.shape[0]
    col = Sx2.shape[1]

    R = np.zeros((row, col))

    for i in range(row):
        for j in range(col):

            R[i, j] = Sx2[i, j] * Sy2[i, j] - Sxy[i, j]**2 - k * (Sx2[i, j] + Sy2[i, j])**2

    Rmax = np.max(R)
    dst_corner = np.zeros((row, col))
    img_with_corner = img.copy()
    for i in range(row):
        for j in range(col):
            if R[i, j] < 0.01 * Rmax:
                R[i, j] = 0

    for i in range(1, row - 1):
        for j in range(1, col - 1):

            if R[i, j] > 0:
                if R[i, j] >= R[i - 1, j - 1] and R[i, j] >= R[i - 1, j] and R[i, j] >= R[i - 1, j + 1] and R[i, j] >= R[i, j - 1] and R[i, j] >= R[i, j + 1] and R[i, j] >= R[i + 1, j - 1] and R[
                        i, j] >= R[i + 1, j] and R[i, j] >= R[i + 1, j + 1]:
                    dst_corner[i, j] = 255
                    img_with_corner = cv.circle(img, (j, i), kernel_size2 // 2, (255, 0, 0), 1)

    return img_with_corner


if __name__ == "__main__":

    # img0 = cv.imread("Lenna.jpg")
    # cv.imshow("img0", img0)

    img0_gray = cv.imread("Lenna_gray.jpg", cv.IMREAD_GRAYSCALE)
    cv.imshow("img0_gray", img0_gray)

    # img0_gray_h = cv.imread("Lenna_gray_h.jpg", cv.IMREAD_GRAYSCALE)
    # cv.imshow("img_gray_h", img0_gray_h)

    # H = np.array([[1, 0, 25], [0, 1, 50], [0, 0, 1]])
    # img1 = geotransform(img0, H, np.array([[img0.shape[0] // 2], [img0.shape[1] // 2], [0]]))
    # cv.imshow("img1", img1.astype(np.uint8))

    # H = np.array([[np.cos(pi / 6), np.sin(pi / 6), 0], [-np.sin(pi / 6), np.cos(pi / 6), 0], [0, 0, 1]])
    # img2 = geotransform(img0, H, np.array([[img0.shape[0] // 2], [img0.shape[1] // 2], [0]]))
    # cv.imshow("img2", img2.astype(np.uint8))

    # H = np.array([[np.cos(pi / 6), np.sin(pi / 6), 50], [-np.sin(pi / 6), np.cos(pi / 6), 25], [0, 0, 1]])
    # img3 = geotransform(img0, H, np.array([[img0.shape[0] // 2], [img0.shape[1] // 2], [0]]))
    # cv.imshow("img3", img3.astype(np.uint8))

    # H = np.array([[1.2 * np.cos(pi / 6), 1.2 * np.sin(pi / 6), 50], [-1.2 * np.sin(pi / 6), 1.2 * np.cos(pi / 6), 25], [0, 0, 1]])
    # img4 = geotransform(img0, H, np.array([[img0.shape[0] // 2], [img0.shape[1] // 2], [0]]))
    # cv.imshow("img4", img4.astype(np.uint8))

    # H = np.array([[1.1, 0.1, 25], [0.2, 1.2, 15], [0, 0, 1]])
    # img5 = geotransform(img0, H, np.array([[img0.shape[0] // 2], [img0.shape[1] // 2], [0]]))
    # cv.imshow("img5", img5.astype(np.uint8))

    # H = np.array([[1.1, 0.1, 25], [0.2, 1.2, 15], [0.001, 0.002, 1.003]])
    # img6 = geotransform(img0, H, np.array([[img0.shape[0] // 2], [img0.shape[1] // 2], [0]]))
    # cv.imshow("img5", img6.astype(np.uint8))

    # img7 = down_sample(img0, 1, 1)
    # cv.imshow("img7", img7.astype(np.uint8))

    # img8_list = gaussian_pyramid(img0, 4, 5, 1)
    # for i in range(len(img8_list)):
    #     cv.imshow("img8_" + str(i), img8_list[i].astype(np.uint8))

    # img9 = up_sample(img0, 5, 1)
    # cv.imshow("img9", img9.astype(np.uint8))

    # img10_list = laplace_pyramid(img0, 5, 5, 1)
    # for i in range(len(img10_list)):
    #     cv.imshow("img10_" + str(i), (img10_list[i] + 128).astype(np.uint8))

    # img11 = build_from_laplace_pyramid(img9_list, 5, 1)
    # cv.imshow("img11", img11.astype(np.uint8))

    # img12_edge = DeriveOG_filter(img0_gray_h, 5, 3)
    # cv.imshow("img12_edge", img12_edge.astype(np.uint8))

    # img13 = NMS(img12_edge)
    # cv.imshow("img13", img13.astype(np.uint8))

    # cv.imwrite("temp.jpg", img13)

    # temp = cv.imread("temp.jpg", cv.IMREAD_GRAYSCALE)
    # cv.imshow("temp", temp.astype(np.uint8))

    # img14 = Hysteresis_thresholding(temp, 85, 43)
    # cv.imshow("img14", img14.astype(np.uint8))

    img15 = Harris_corner(img0_gray, 5, 1, 5, 1, 0.04)
    cv.imshow("img15", img15.astype(np.uint8))

    cv.waitKey(0)
    cv.destroyAllWindows()
