from cv2 import cv2 as cv
import random
import numpy as np


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


def bilateral_filter(img, kernel_size, sigmas, sigmar):
    img_dim = len(img.shape)

    if img_dim == 3:
        img_row = img.shape[0]
        img_col = img.shape[1]
        img_channel = img.shape[2]

        kernels = gaussian_kernel(kernel_size, sigmas)
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


# 添加椒盐噪声，k表示一共有k个白点和k个黑点
def addSaltNoise(img, k):
    dst_img = img.copy()
    img_dim = len(img.shape)
    w = img.shape[0]
    h = img.shape[1]
    for k in range(k):
        i = random.randint(0, w - 1)
        j = random.randint(0, h - 1)
        if img_dim == 1:
            dst_img[i, j] = 0
        if img_dim == 3:
            dst_img[i, j, 0] = 0
            dst_img[i, j, 1] = 0
            dst_img[i, j, 2] = 0
    for k in range(k):
        i = random.randint(0, w - 1)
        j = random.randint(0, h - 1)
        if img_dim == 1:
            dst_img[i, j] = 255
        if img_dim == 3:
            dst_img[i, j, 0] = 255
            dst_img[i, j, 1] = 255
            dst_img[i, j, 2] = 255
    return dst_img


# 添加均值为mu标准差为sigma的高斯噪声
def addGaussNoise(img, mu, sigma):
    noise = np.random.normal(mu, sigma, img.shape)
    # dst_img = np.clip(img, -1, 1)
    dst_img = img.copy()
    dst_img = dst_img + noise
    dst_img = cv.convertScaleAbs(dst_img)
    return dst_img


# 常数核卷积，边缘填充为0，卷积核为正方形
def conv(img, kernel):
    img_dim = len(img.shape)

    if img_dim == 3:
        img_row = img.shape[0]
        img_col = img.shape[1]
        img_channel = img.shape[2]

        kernel_size = kernel.shape[0]
        result = np.zeros((img_row, img_col, img_channel))
        ex_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), 'constant', constant_values=(1, 1))
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
        ex_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), 'constant', constant_values=(1, 1))
        result_row = result.shape[0]
        result_col = result.shape[1]
        for i in range(result_row):
            for j in range(result_col):
                result[i, j] = np.sum(ex_img[i:i + kernel_size, j:j + kernel_size] * kernel)
                
    result = cv.convertScaleAbs(result)
    return result


# 转换为灰度图
def gray(img):
    row, col, _ = img.shape
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            result[i, j] = 0.114 * img[i, j, 0] + 0.299 * img[i, j, 1] + 0.587 * img[i, j, 2]
    result = cv.convertScaleAbs(result)
    return result


# 均匀算子
def mean_filter(img):
    kernel = np.zeros((3, 3), np.float32)
    for i in range(0, 3, 1):
        for j in range(0, 3, 1):
            kernel[i, j] = 1
    kernel = kernel / 9
    result = conv(img, kernel)

    result = cv.convertScaleAbs(result)
    return result


# 边缘检测，这里是卷积以后进行非线性转换
def edge_filter(img):
    kernel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    result1 = conv(img, kernel1)
    result1 = cv.convertScaleAbs(result1)
    result2 = conv(img, kernel2)
    result2 = cv.convertScaleAbs(result2)
    result = 0.5 * result1 + 0.5 * result2

    result = cv.convertScaleAbs(result)
    return result


# 锐化
def sharpen_filter(img):
    kernel1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    alpha = 0.8
    kernel = (1 + alpha) * kernel1 - alpha * kernel2
    result = conv(img, kernel)

    result = cv.convertScaleAbs(result)
    return result


# 中值滤波,用kernelz_size*kernel_size大小的卷积窗来选取中值
def median_filter(img, kernel_size):
    img_dim = len(img.shape)

    if img_dim == 3:
        img_row = img.shape[0]
        img_col = img.shape[1]
        img_channel = img.shape[2]

        result = np.zeros((img_row, img_col, img_channel))
        ex_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), 'constant', constant_values=(1, 1))
        result_row = result.shape[0]
        result_col = result.shape[1]
        result_channel = result.shape[2]
        for i in range(result_row):
            for j in range(result_col):
                for k in range(result_channel):
                    result[i, j, k] = np.median(ex_img[i:i + kernel_size, j:j + kernel_size, k])

    elif img_dim == 2:
        img_row = img.shape[0]
        img_col = img.shape[1]

        result = np.zeros((img_row, img_col))
        ex_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), 'constant', constant_values=(1, 1))
        result_row = result.shape[0]
        result_col = result.shape[1]
        for i in range(result_row):
            for j in range(result_col):
                result[i, j] = np.median(ex_img[i:i + kernel_size, j:j + kernel_size])

    result = cv.convertScaleAbs(result)
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
def gauss_filter(img, kernel_size, sigma):

    kernel = gaussian_kernel(kernel_size, sigma)
    result = conv(img, kernel)
    result = cv.convertScaleAbs(result)
    return result.astype(np.uint8)


if __name__ == "__main__":
    '''
        img0 = cv.imread("Lenna.jpg")
        img1 = mean_filter(img0)
        img2 = edge_filter(img0)
        img3 = sharpen_filter(img0)
        cv.imshow("raw image", img0.astype("uint8"))
        cv.imshow("mean filter", img1.astype("uint8"))
        cv.imshow("edge filter", img2.astype("uint8"))
        cv.imshow("sharpening filter", img3.astype("uint8"))
    '''

    # img_salt = cv.imread("Lenna_salt.jpg")
    img_gauss = cv.imread("Lenna_gauss.jpg")
    # img4 = median_filter(img_salt, 3)
    img5 = gauss_filter(img_gauss, 5, 2)
    # cv.imshow("img_salt", img_salt)
    # cv.imshow("img4", img4.astype("uint8"))
    cv.imshow("img_gauss", img_gauss)
    cv.imshow("img5", img5.astype("uint8"))
    cv.waitKey(0)
    cv.destroyAllWindows()
