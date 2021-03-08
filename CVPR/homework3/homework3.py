from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import random

epsilon = 1e-9


def my_findHomography(right_kps, left_kps):

    length = len(right_kps)

    M = np.zeros((2 * length, 9))
    b = np.zeros((2 * length, 0))
    for i in range(length):
        right_kp = right_kps[i]
        left_kp = left_kps[i]
        M[2 * i, :] = np.array([right_kp[0], right_kp[1], 1, 0, 0, 0, -left_kp[0] * right_kp[0], -left_kp[0] * right_kp[1], -left_kp[0]])
        M[2 * i + 1, :] = np.array([0, 0, 0, right_kp[0], right_kp[1], 1, -left_kp[1] * right_kp[0], -left_kp[1] * right_kp[1], -left_kp[1]])
    U, S, V = svd(M)
    h = V.T[:, 8]
    h = h / (h[8] + epsilon)
    H = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]])
    return H


def count_inliers(right_kps, left_kps, H, threshold):

    length = len(right_kps)
    inliers = 0
    outliers = 0
    for i in range(length):
        distance = my_SSD(right_kps[i], left_kps[i], H)
        if distance < threshold:
            inliers = inliers + 1
        else:
            outliers = outliers + 1
    return inliers, outliers


def my_SSD(right_kp, left_kp, H):

    right_homogeneous = np.array([[right_kp[0]], [right_kp[1]], [1]])
    left_homogeneous = np.array([[left_kp[0]], [left_kp[1]], [1]])
    left_predict = np.dot(H, right_homogeneous)
    left_predict = left_predict / (left_predict[2] + epsilon)
    distance = np.sum((left_predict - left_homogeneous)**2)
    return distance


def my_RANSAC(right_kps, left_kps, threshold):

    length = len(right_kps)

    temp_left_kps = np.zeros((4, 2))
    temp_right_kps = np.zeros((4, 2))

    max_inliers = -1
    good_H = np.zeros((3, 3))

    # 循环迭代100次，找到最佳的H
    for j in range(100):
        for i in range(4):
            rand_num = random.randint(0, length - 1)
            temp_left_kps[i, :] = left_kps[rand_num, :]
            temp_right_kps[i, :] = right_kps[rand_num, :]
        H = my_findHomography(temp_right_kps, temp_left_kps)
        inliers, outliers = count_inliers(right_kps, left_kps, H, threshold)
        if inliers > max_inliers:
            max_inliers = inliers
            good_H = H

    return good_H


if __name__ == "__main__":

    img_left = cv2.imread('left.png', 1)
    img_right = cv2.imread('right.png', 1)

    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_left_des = cv2.SIFT_create()
    left_kps, left_features = img_left_des.detectAndCompute(img_left_gray, None)

    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    img_right_des = cv2.SIFT_create()
    right_kps, right_features = img_right_des.detectAndCompute(img_right_gray, None)

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(left_features, right_features, 2)
    print("raw_matches: {0}".format(len(raw_matches)))

    ratio = 0.5
    threshold = 1
    matches = []
    good = []
    for m in raw_matches:
        # print(m[0].distance, m[1].distance)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            good.append([m[0]])
            matches.append((m[0].queryIdx, m[0].trainIdx))

    # 显示特征匹配结果
    img = cv2.drawMatchesKnn(img_left, left_kps, img_right, right_kps, good[:30], outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.title('Feature matching')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    left_kps = np.float32([kp.pt for kp in left_kps])
    right_kps = np.float32([kp.pt for kp in right_kps])

    print("good matches: {0}".format(len(matches)))

    if (len(matches) > 4):
        left_kps = np.float32([left_kps[i] for (i, _) in matches])
        right_kps = np.float32([right_kps[i] for (_, i) in matches])
        # H, status = cv2.findHomography(right_kps, left_kps, cv2.RANSAC, threshold)
        # H = my_findHomography(right_kps, left_kps)
        H = my_RANSAC(right_kps, left_kps, 1)
    else:
        print("not enough matching!")

    left_h, left_w = img_left.shape[:2]
    right_h, right_w = img_right.shape[:2]
    image = np.zeros((max(left_h, right_h), left_w + right_w, 3), dtype='uint8')
    image[0:img_right.shape[0], 0:img_right.shape[1]] = img_right
    image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    image[0:img_left.shape[0], 0:img_left.shape[1]] = img_left

    plt.figure()
    plt.title('Image stitching')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()