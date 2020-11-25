from cv2 import cv2 as cv
import random
import numpy as np


def padding(img, ud, lr, paddingtype):
    img_dim = len(img.shape)

    if img_dim == 3:
        up = ud[0]
        down = ud[1]
        left = lr[0]
        right = lr[1]

        dst_img = np.zeros((img.shape[0] + up + down, img.shape[1] + left + right, 3))
        dst_img[up:up + img.shape[0], left:left + img.shape[1], 0:3] = img

        if paddingtype == "black":
            pass
        if paddingtype == "wrap":
            dst_img[0:up, 0:left, 0:3] = dst_img[img.shape[0]:img.shape[0] + up, img.shape[1]:img.shape[1] + left, 0:3]
            dst_img[up:up + img.shape[0], 0:left, 0:3] = dst_img[up:up + img.shape[0], img.shape[1]:img.shape[1] + left, 0:3]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, 0:left, 0:3] = dst_img[up:up + down, img.shape[1]:img.shape[1] + left, 0:3]
            dst_img[0:up, left:left + img.shape[1], 0:3] = dst_img[img.shape[0]:img.shape[0] + up, left:left + img.shape[1], 0:3]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left:left + img.shape[1], 0:3] = dst_img[up:up + down, left:left + img.shape[1], 0:3]
            dst_img[0:up, left + img.shape[1]:left + img.shape[1] + right, 0:3] = dst_img[img.shape[0]:img.shape[0] + up, left:left + right, 0:3]
            dst_img[up:up + img.shape[0], left + img.shape[1]:left + img.shape[1] + right, 0:3] = dst_img[up:up + img.shape[0], left:left + right, 0:3]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left + img.shape[1]:left + img.shape[1] + right, 0:3] = dst_img[up:up + down, left:left + right, 0:3]
        if paddingtype == "copy":
            dst_img[0:up, 0:left, 0:3] = dst_img[up:up + 1, left:left + 1, 0:3]
            dst_img[up:up + img.shape[0], 0:left, 0:3] = dst_img[up:up + img.shape[0], left:left + 1, 0:3]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, 0:left, 0:3] = dst_img[up + img.shape[0] - 1:up + img.shape[0], left:left + 1, 0:3]
            dst_img[0:up, left:left + img.shape[1], 0:3] = dst_img[up:up + 1, left:left + img.shape[1], 0:3]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left:left + img.shape[1], 0:3] = dst_img[up + img.shape[0] - 1:up + img.shape[0], left:left + img.shape[1], 0:3]
            dst_img[0:up, left + img.shape[1]:left + img.shape[1] + right, 0:3] = dst_img[up:up + 1, left + img.shape[1] - 1:left + img.shape[1], 0:3]
            dst_img[up:up + img.shape[0], left + img.shape[1]:left + img.shape[1] + right, 0:3] = dst_img[up:up + img.shape[0], left + img.shape[1] - 1:left + img.shape[1], 0:3]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left + img.shape[1]:left + img.shape[1] + right, 0:3] = dst_img[up + img.shape[0] - 1:up + img.shape[0], left + img.shape[1] - 1:left +
                                                                                                                                img.shape[1], 0:3]
        if paddingtype == "reflect":
            dst_img[0:up, 0:left, 0:3] = np.flip(dst_img[up:up + up, left:left + left, 0:3], (0, 1))
            dst_img[up:up + img.shape[0], 0:left, 0:3] = np.flip(dst_img[up:up + img.shape[0], left:left + left, 0:3], 1)
            dst_img[up + img.shape[0]:up + img.shape[0] + down, 0:left, 0:3] = np.flip(dst_img[up + img.shape[0] - down:up + img.shape[0], left:left + left, 0:3], (0, 1))
            dst_img[0:up, left:left + img.shape[1], 0:3] = np.flip(dst_img[up:up + up, left:left + img.shape[1], 0:3], 0)
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left:left + img.shape[1], 0:3] = np.flip(dst_img[up + img.shape[0] - down:up + img.shape[0], left:left + img.shape[1], 0:3], 0)
            dst_img[0:up, left + img.shape[1]:left + img.shape[1] + right, 0:3] = np.flip(dst_img[up:up + up, left + img.shape[1] - right:left + img.shape[1], 0:3], (0, 1))
            dst_img[up:up + img.shape[0], left + img.shape[1]:left + img.shape[1] + right, 0:3] = np.flip(dst_img[up:up + img.shape[0], left + img.shape[1] - right:left + img.shape[1]], 1)
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left + img.shape[1]:left + img.shape[1] + right, 0:3] = np.flip(
                dst_img[up + img.shape[0] - down:up + img.shape[0], left + img.shape[1] - right:left + img.shape[1], 0:3], (0, 1))

    if img_dim == 2:
        up = ud[0]
        down = ud[1]
        left = lr[0]
        right = lr[1]

        dst_img = np.zeros((img.shape[0] + up + down, img.shape[1] + left + right))
        dst_img[up:up + img.shape[0], left:left + img.shape[1]] = img

        if paddingtype == "black":
            pass
        if paddingtype == "wrap":
            dst_img[0:up, 0:left] = dst_img[img.shape[0]:img.shape[0] + up, img.shape[1]:img.shape[1] + left]
            dst_img[up:up + img.shape[0], 0:left] = dst_img[up:up + img.shape[0], img.shape[1]:img.shape[1] + left]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, 0:left] = dst_img[up:up + down, img.shape[1]:img.shape[1] + left]
            dst_img[0:up, left:left + img.shape[1]] = dst_img[img.shape[0]:img.shape[0] + up, left:left + img.shape[1]]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left:left + img.shape[1]] = dst_img[up:up + down, left:left + img.shape[1]]
            dst_img[0:up, left + img.shape[1]:left + img.shape[1] + right] = dst_img[img.shape[0]:img.shape[0] + up, left:left + right]
            dst_img[up:up + img.shape[0], left + img.shape[1]:left + img.shape[1] + right] = dst_img[up:up + img.shape[0], left:left + right]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left + img.shape[1]:left + img.shape[1] + right] = dst_img[up:up + down, left:left + right]
        if paddingtype == "copy":
            dst_img[0:up, 0:left] = dst_img[up:up + 1, left:left + 1]
            dst_img[up:up + img.shape[0], 0:left] = dst_img[up:up + img.shape[0], left:left + 1]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, 0:left] = dst_img[up + img.shape[0] - 1:up + img.shape[0], left:left + 1]
            dst_img[0:up, left:left + img.shape[1]] = dst_img[up:up + 1, left:left + img.shape[1]]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left:left + img.shape[1]] = dst_img[up + img.shape[0] - 1:up + img.shape[0], left:left + img.shape[1]]
            dst_img[0:up, left + img.shape[1]:left + img.shape[1] + right] = dst_img[up:up + 1, left + img.shape[1] - 1:left + img.shape[1]]
            dst_img[up:up + img.shape[0], left + img.shape[1]:left + img.shape[1] + right] = dst_img[up:up + img.shape[0], left + img.shape[1] - 1:left + img.shape[1]]
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left + img.shape[1]:left + img.shape[1] + right] = dst_img[up + img.shape[0] - 1:up + img.shape[0], left + img.shape[1] - 1:left +
                                                                                                                           img.shape[1]]
        if paddingtype == "reflect":
            dst_img[0:up, 0:left] = np.flip(dst_img[up:up + up, left:left + left], (0, 1))
            dst_img[up:up + img.shape[0], 0:left] = np.flip(dst_img[up:up + img.shape[0], left:left + left], 1)
            dst_img[up + img.shape[0]:up + img.shape[0] + down, 0:left] = np.flip(dst_img[up + img.shape[0] - down:up + img.shape[0], left:left + left], (0, 1))
            dst_img[0:up, left:left + img.shape[1]] = np.flip(dst_img[up:up + up, left:left + img.shape[1]], 0)
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left:left + img.shape[1]] = np.flip(dst_img[up + img.shape[0] - down:up + img.shape[0], left:left + img.shape[1]], 0)
            dst_img[0:up, left + img.shape[1]:left + img.shape[1] + right] = np.flip(dst_img[up:up + up, left + img.shape[1] - right:left + img.shape[1]], (0, 1))
            dst_img[up:up + img.shape[0], left + img.shape[1]:left + img.shape[1] + right] = np.flip(dst_img[up:up + img.shape[0], left + img.shape[1] - right:left + img.shape[1]], 1)
            dst_img[up + img.shape[0]:up + img.shape[0] + down, left + img.shape[1]:left + img.shape[1] + right] = np.flip(
                dst_img[up + img.shape[0] - down:up + img.shape[0], left + img.shape[1] - right:left + img.shape[1]], (0, 1))

        return dst_img


# 常数核卷积，边缘填充为0，卷积核为正方形
def conv(img, kernel, paddingtype):
    img_dim = len(img.shape)

    if img_dim == 3:
        img_row = img.shape[0]
        img_col = img.shape[1]
        img_channel = img.shape[2]

        kernel_size = kernel.shape[0]
        result = np.zeros((img_row, img_col, img_channel))
        ex_img = padding(img, (kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), paddingtype)
        result_row = result.shape[0]
        result_col = result.shape[1]
        result_channel = result.shape[2]
        for i in range(result_row):
            for j in range(result_col):
                for k in range(result_channel):
                    result[i, j, k] = np.sum(ex_img[i:i + kernel_size, j:j + kernel_size, k] * kernel)

    elif img_dim == 2:
        img_row = img.shape[0]
        img_col = img.shape[1]
        kernel_size = kernel.shape[0]
        result = np.zeros((img_row, img_col))
        ex_img = padding(img, (kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), paddingtype)
        result_row = result.shape[0]
        result_col = result.shape[1]
        for i in range(result_row):
            for j in range(result_col):
                result[i, j] = np.sum(ex_img[i:i + kernel_size, j:j + kernel_size] * kernel)

    return result


def gaussian_kernel(kernel_size, sigma):
    gauss_k = np.zeros((kernel_size, kernel_size), np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            norm = (i - kernel_size // 2)**2 + (j - kernel_size // 2)**2
            gauss_k[i, j] = np.exp(-norm / (sigma**2 * 2))
    s = gauss_k.sum()
    gauss_k = gauss_k / s
    return gauss_k


# 高斯滤波
def gauss_filter(img, kernel_size, sigma, paddingtype="reflect"):

    kernel = gaussian_kernel(kernel_size, sigma)
    result = conv(img, kernel, paddingtype)

    return result


if __name__ == "__main__":

    img0 = cv.imread("Lenna.jpg")
    cv.imshow("raw image", img0)

    cv.waitKey(0)
    cv.destroyAllWindows()
