import numpy as np
from ransac import *
import cv2
from Affine import *

'''
    把变化后的图像对齐到原始图像，完美配准
'''


# 提取关键点和关联的描述子
def extract_SIFT(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # >>>版权专利问题
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndComputer(img_gray, None)
    kp = np.array([p.pt for p in kp]).T
    return kp, desc


# 获取找到的所有关键点的位置
def match_SIFT(descriptor_source, descriptor_target):
    # 寻找在所有匹配的描述子中找到最佳的两个匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor_source, descriptor_target, k=2)
    # 创建一个空的Numpy数组存放关键点的信息
    pos = np.array([], dtype=np.int32).reshape((0, 2))
    matches_num = len(matches)
    # queryIdx返回目标图像中的描述子下标 ， trainIdx返回源图像中的描述子下标，存放在temp中
    for i in range(matches_num):
        if matches[i][0].distance <= 0.8 * matches[i][1].distance :
            temp = np.array([matches[i][0].queryIdx, matches[i][0].trainIdx])
            pos = np.vstack((pos, temp))
    return pos


# 获得描述子位置信息后，获得图像配准中的单应性矩阵
# s:source  t:target
def affine_matrix(s, t, pos):
    # 从pos中提取最佳描述子位置，保存在s,t 中
    s = s[:, pos[:, 0]]
    t = t[:, pos[:, 1]]
    # 利用ransac_fit函数求出内点，其中内点是两张图象中呈现出最大相似性
    _, _, inliers = ransac_fit(s, t)
    # 用inliers内点的下标，提取最佳的来源图像的关键点和目标图像关键点
    s = s[:, inliers[0]]
    t = t[:, inliers[0]]
    # 利用estimate_affine函数得到最终的矩阵
    A, t = estimate_affine(s, t)
    # 将A，t水平并置 并将整体作为  单应性(相似性)矩阵  返回
    M = np.hsplit((A, t))
    return M

