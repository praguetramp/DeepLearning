import numpy as np
import cv2
from Affine import *
from ransac import *
from Align import *

# 读取源图像，目标图像
img_source = cv2.imread("laifushi1.jpg")
img_target = cv2.imread("laifushi2.jpg")
# 获取源图像、目标图像的关键点信息和描述子
keypoint_source, descriptor_source = extract_SIFT(img_source)
keypoint_target, descriptor_target = extract_SIFT(img_target)
# 获取关键点的位置信息
pos = match_SIFT(descriptor_source, descriptor_target)
# 求得图像配准的 最佳单应性矩阵
H = affine_matrix(keypoint_source, keypoint_target, pos)

# 图像配准，首先获取目标图像的行数、列数
rows, cols, _ = img_target.shape
# 在来源图像上应用单应性矩阵，并缩放为目标大小
warp = cv2.warpAffine(img_source, H, (cols, rows))
# 将两张图像融合，源、目标
merge = np.uint8(img_target * 0.5 + warp * 0.5)
# 输出显示
cv2.imshow("img", merge)
cv2.waitKey()
cv2.destroyAllWindows()