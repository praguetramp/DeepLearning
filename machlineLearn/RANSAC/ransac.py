# 图像配准：将一张图像准确叠加到另一张图像的相同地方的过程
# 找到两张图像之间的单应性（相似性）
'''过程：
    1.特征检测和抽取
 ![](laifushi2.jpg)   2.特征匹配
    3.转换函数拟合  （RANSAC算法用于在该步骤找到转换函数）
    4.图像转换和图像重采样
'''

'''
    （1）随机找到两张图片的共同特征点，求出单应性矩阵
    （2）重复多次，直到找到具有内点(inlier)个数最多的单应性矩阵
'''
import numpy as np
from Affine import *
k = 3
threshold = 1
ITER_NUM = 2000

# 确定模型中存在的误差，并确保生成的仿射矩阵和匹配的描述子的误差尽可能小
def residual_lengths(x, y, s, t):
    # 仿射矩阵x 和 源图像关键点矩阵的线性模型，给出目标图像的  点估计
    e = np.dot(x, s)+y
    # 目标点 与 目标点的估计相减，先求平方，在所有求和，再所有整体开方
    # 消除负值的影响，均 >>>方根误差估计,残差估计
    diff_square = np.power(e-t, 2)
    residual = np.sqrt(np.sum(diff_square, axis=0))
    # 返回残差
    return residual

# 获取关键点


def ransac_fit (pts_s, pts_t):
    # 内点的数量
    inliers_num = 0
    # 仿射矩阵
    A = None
    t = None
    # 内点下标
    inliers = None

    # 重复2000次，得到最佳矩阵(内点数量最多)
    for i in range(ITER_NUM):
        # 用于求出仿射变换的临时矩阵,在这里需要随机生成坐标
        idx = np.random.randint(0, pts_s.shape[1], (k, 1))
        A_tmp, t_tmp = estimate_affine(pts_s.shape[:, idx], pts_t.shape[:, idx])
        # 将临时函数用于求误差，帮助我我们确定最后的矩阵
        residual = residual_lengths(A_tmp, t_tmp, pts_s, pts_t)

        if not(residual is None):
            # 知道残差/误差之后，可以和阈值作比较，将其计算到(内点的)实例里面
            inliers_tmp = np.where(residual < threshold)

            inliers_num_tmp = len(inliers_tmp[0])
            # 比较当前求得的内点和全局维护的内点，如果内点数增加，则更新内点数量、更新仿射矩阵A和t，并保存内点下标
            if inliers_num_tmp > inliers_num:
                # 更新内点数量
                inliers_num = inliers_num_tmp
                # 更新内点下标
                inliers = inliers_num
                # 更新仿射矩阵
                A = A_tmp
                t = t_tmp
            else:
                pass
    return A, t, inliers


