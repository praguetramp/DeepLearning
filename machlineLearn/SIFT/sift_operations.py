import cv2
import numpy as np
from matplotlib import pyplot as plt

# SIFT调用


def extract_sift_features(img):
    # 将SIFT算法保存在sift_init 变量中，方便调用 >>>该算法为专利，该版本不能使用
    sift_init = cv2.xfeatures2d.SIFT_create()
    # 利用其中的函数返回图像的关键点、描述信息
    key_points, descriptors = sift_init.detectAndCompute(img, None)
    return key_points, descriptors
# 显示特征，输出关键点 和 相似性


def showing_sift_features(img1, img2, key_points):
    # drawKeyPoints 在两张图像中找到关键点
    return plt.imshow(cv2.drawKeypoints(img1, key_points, img2.copy()))
