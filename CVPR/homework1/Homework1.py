####################
# 滤波实验
# 2D高斯模板的设计（给定方差生成滤波核）；图像的高斯滤波；在高斯滤波中不同边界处理方法实验；
# 高斯核与高斯核的卷积实验；利用两个相同方差的一维行列高斯核卷积生成2D高斯核，利用一维行列高斯对图像进行滤波；不同方差高斯核之差对图像进行滤波
# 利用两个高斯核设计图像锐化滤波器核；
# 图像的双边滤波实验
# 图像的Fourier变换，显示幅度谱核相位谱；利用高斯滤波器进行图像的频率域滤波
####################

import numpy as np
from cv2 import cv2 as cv
import random


def padding(img, ud, lr, paddingtype):
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

    dst_img = cv.convertScaleAbs(dst_img)
    return dst_img


# 常数核卷积，边缘填充为0，卷积核为正方形
def conv2(img, kernel):
    img_dim = len(img.shape)

    if img_dim == 3:
        img_row = img.shape[0]
        img_col = img.shape[1]
        img_channel = img.shape[2]

        kernel_size = kernel.shape[0]
        result = np.zeros((img_row, img_col, img_channel))
        # ex_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), 'constant', constant_values=(0, 0))
        ex_img = padding(img, (kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), "reflect")
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
        # ex_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), 'constant', constant_values=(0, 0))
        ex_img = padding(img, (kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), "black")
        result_row = result.shape[0]
        result_col = result.shape[1]
        for i in range(result_row):
            for j in range(result_col):
                result[i, j] = np.sum(ex_img[i:i + kernel_size, j:j + kernel_size] * kernel)
    result = cv.convertScaleAbs(result)
    return result


# 2D高斯模板的设计（给定方差生成滤波核）
def gaussian_kernel2(kernel_size, sigma):
    gauss_k = np.zeros((kernel_size, kernel_size), np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            norm = (i - kernel_size // 2)**2 + (j - kernel_size // 2)**2
            gauss_k[i, j] = np.exp(-norm / (sigma**2 * 2))
    s = gauss_k.sum()
    gauss_k = gauss_k / s
    return gauss_k


# 图像的高斯滤波
def gauss_filter2(img, kernel_size, sigma):

    kernel = gaussian_kernel2(kernel_size, sigma)
    result = conv2(img, kernel)
    result = cv.convertScaleAbs(result)
    return result.astype(np.uint8)


def conv1_col(img, kernel):
    img_dim = len(img.shape)

    if img_dim == 3:
        img_row = img.shape[0]
        img_col = img.shape[1]
        img_channel = img.shape[2]
        kernel_size = kernel.shape[0]
        result = np.zeros((img_row, img_col, img_channel))
        # 竖向卷积
        ex_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
        result_row = result.shape[0]
        result_col = result.shape[1]
        result_channel = result.shape[2]
        for i in range(result_row):
            for j in range(result_col):
                for k in range(result_channel):
                    result[i, j, k] = np.sum(ex_img[i:i + kernel_size, j:j + 1, k] * kernel)

    elif img_dim == 2:
        img_row = img.shape[0]
        img_col = img.shape[1]
        kernel_size = kernel.shape[0]

        result = np.zeros((img_row, img_col))
        # 竖向卷积
        ex_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'constant', constant_values=(0, 0))
        result_row = result.shape[0]
        result_col = result.shape[1]
        for i in range(result_row):
            for j in range(result_col):
                result[i, j] = np.sum(ex_img[i:i + kernel_size, j:j + 1] * kernel)

    result = cv.convertScaleAbs(result)
    return result


def conv1_row(img, kernel):
    img_dim = len(img.shape)

    if img_dim == 3:
        img_row = img.shape[0]
        img_col = img.shape[1]
        img_channel = img.shape[2]
        kernel_size = kernel.shape[1]

        result = np.zeros((img_row, img_col, img_channel))
        # 横向卷积
        ex_img = np.pad(img, ((0, 0), (kernel_size // 2, kernel_size // 2), (0, 0)), 'constant', constant_values=(0, 0))
        result_row = result.shape[0]
        result_col = result.shape[1]
        result_channel = result.shape[2]
        for i in range(result_row):
            for j in range(result_col):
                for k in range(result_channel):
                    result[i, j, k] = np.sum(ex_img[i:i + 1, j:j + kernel_size, k] * kernel)

    elif img_dim == 2:
        img_row = img.shape[0]
        img_col = img.shape[1]
        kernel_size = kernel.shape[1]

        result = np.zeros((img_row, img_col))
        # 横向卷积
        ex_img = np.pad(img, ((0, 0), (kernel_size // 2, kernel_size // 2)), 'constant', constant_values=(0, 0))
        result_row = result.shape[0]
        result_col = result.shape[1]
        for i in range(result_row):
            for j in range(result_col):
                result[i, j] = np.sum(ex_img[i:i + 1, j:j + kernel_size // 2] * kernel)

    result = cv.convertScaleAbs(result)
    return result


def gaussian_kernel1(kernel_size, sigma):
    gauss_k = np.zeros((kernel_size, 1))
    for i in range(kernel_size):
        norm = (i - kernel_size // 2)**2
        gauss_k[i, 0] = np.exp(-norm / (sigma**2 * 2))
    s = gauss_k.sum()
    gauss_k = gauss_k / s
    return gauss_k


def gauss_filter1(img, kernel_size, sigma):

    kernel = gaussian_kernel1(kernel_size, sigma)
    result = conv1_col(img, kernel)
    result = conv1_row(result, kernel.transpose())
    result = cv.convertScaleAbs(result)
    return result.astype(np.uint8)


def DoG_filter(img, kernel_size, sigma1, sigma2):
    kernel1 = gaussian_kernel2(kernel_size, sigma1)
    kernel2 = gaussian_kernel2(kernel_size, sigma2)
    result = conv2(img, kernel1 - kernel2)
    result = cv.convertScaleAbs(result)
    return result.astype(np.uint8)


def sharpen_filter(img, kernel_size, sigma, alpha):
    kernel1 = np.zeros((kernel_size, kernel_size))
    kernel1[kernel_size // 2, kernel_size // 2] = 1
    kernel2 = gaussian_kernel2(kernel_size, sigma)
    kernel = (1 + alpha) * kernel1 - alpha * kernel2
    result = conv2(img, kernel)

    result = cv.convertScaleAbs(result)
    return result.astype(np.uint8)


def bilateral_filter(img, kernel_size, sigmas, sigmar):
    img_dim = len(img.shape)

    if img_dim == 3:
        img_row = img.shape[0]
        img_col = img.shape[1]
        img_channel = img.shape[2]

        kernels = gaussian_kernel2(kernel_size, sigmas)
        kernelr = np.zeros((kernel_size, kernel_size))
        kernel = np.zeros((kernel_size, kernel_size))

        result = np.zeros((img_row, img_col, img_channel))
        # ex_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), 'constant', constant_values=(0, 0))
        ex_img = padding(img, (kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), "reflect")
        result_row = result.shape[0]
        result_col = result.shape[1]
        result_channel = result.shape[2]
        for i in range(result_row):
            for j in range(result_col):

                for m in range(0, kernel_size, 1):
                    for n in range(0, kernel_size, 1):
                        kernelr[m, n] = (float(ex_img[i + m, j + n, 0]) - float(ex_img[i + kernel_size // 2, j + kernel_size // 2, 0]))**2 + (float(ex_img[i + m, j + n, 1]) - float(
                            ex_img[i + kernel_size // 2, j + kernel_size // 2, 1]))**2 + (float(ex_img[i + m, j + n, 2]) - float(ex_img[i + kernel_size // 2, j + kernel_size // 2, 2]))**2
                kernelr = np.exp(-kernelr / (2 * sigmar**2))
                kernel = kernels * kernelr
                s = np.sum(kernel)
                kernel = kernel / s
                for k in range(result_channel):
                    result[i, j, k] = np.sum(ex_img[i:i + kernel_size, j:j + kernel_size, k] * kernel)

    result = cv.convertScaleAbs(result)
    return result.astype(np.uint8)


def FFT(img):
    img_dim = len(img.shape)
    if img_dim == 2:
        M = img.shape[0]
        N = img.shape[1]

        X = np.zeros((M, N))
        for k in range(M):
            for l in range(N):
                X[k, l] = 0
                print(1)
                for m in range(M):
                    for n in range(N):
                        X[k, l] += img[m, n] * np.exp(complex(0, -k * 2 * np.pi * m / M - l * 2 * np.pi * n / N))
        X = X / (M * N)
        return X


if __name__ == "__main__":

    img0 = cv.imread("Lenna.jpg")
    cv.imshow("img0", img0)
    
    img1 = padding(img0, (20, 20), (20, 20), "black")
    cv.imshow("img1", img1)

    img2 = padding(img0, (20, 20), (20, 20), "wrap")
    cv.imshow("img2", img2)

    img3 = padding(img0, (20, 20), (20, 20), "copy")
    cv.imshow("img3", img3)

    img4 = padding(img0, (20, 20), (20, 20), "reflect")
    cv.imshow("img4", img4)

    img5 = gauss_filter2(img0, 5, 12)
    cv.imshow("img5", img5)

    img6 = gauss_filter1(img0, 5, 12)
    cv.imshow("img6", img6)

    img7 = DoG_filter(img0, 11, 2, 12)
    cv.imshow("img7", img7)

    img8 = sharpen_filter(img0, 5, 2, 0.8)
    cv.imshow("img8", img8)

    img0_gauss = cv.imread("Lenna_gauss.jpg")
    cv.imshow("img0_gauss", img0_gauss)
    img9 = bilateral_filter(img0_gauss, 5, 12, 36)
    cv.imshow("img9", img9)

    cv.waitKey(0)
    cv.destroyAllWindows()
