# SIFT 尺度不变特征转移
# 作用：特征映射，关键点的连接
# 特点：不管图像放大、缩小、旋转都不会影响我们找到图像间的相似性
'''
过程：
    1.构造尺度不变的空间
    2.求两个高斯之差
    3.找出图像的关键点
    4.为了高效地比较，移除非关键点
    5.提供步骤（3）中找到的关键点的方向
    6.确定唯一关键特征
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sift_operations import *
print("Make sure your pictures are in the same directory !")
x = input("Enter the first image's name :")
img1 = cv2.imread(x)
y = input("Enter the second image's name :")
img2 = cv2.imread(y)
# convert to gray for this algorithm
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# extract img  key and info
img1_key_points, img1_descriptors = extract_sift_features(img1_gray)
img2_key_points, img2_descriptors = extract_sift_features(img2_gray)

print("Displaying sift features :")
showing_sift_features(img1_gray, img1, img1_key_points)

# 计算曼哈顿距离，沿着网格的距离
norm = cv2.NORM_L2
# 利用曼哈顿距离匹配关键点的描述子
bruteForce = cv2.BFMatcher(norm)
# 匹配了两个子描述，将结果matches根据曼哈顿距离排序
matches = bruteForce.match(img1_descriptors, img2_descriptors)
matches = sorted(matches, key=lambda match: match.distance)

# 根据排完序的匹配结果的前100条记录，连接两张图像的关键点
matched_img = cv2.drawMatches(img1, img1_key_points, img2, img2_key_points,
                              matches[:100], img2.copy())
# 最后显示的图像
plt.figure(figsize=(100, 300))
plt.imshow(matched_img)
