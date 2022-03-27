import numpy as np
'''
    将图像旋转、平移和缩放用来做对比
'''

# 以来源图像和目标图像的关键点矩阵作为输入
# 返回反射变换矩阵


def estimate_affine(s, t):
    # 基于 来源关键点矩阵 的维度，初始化一个临时矩阵，往其中填充来源关键点
    num = s.shape[1]
    M = np.zeros((2*num, 6))

    for i in range(num):
        temp = [[s[0, i], s[1, i], 0, 0, 1, 0], [0, 0, s[0, i], s[1, i], 0, 1]]
        M[2*i: 2*i+2:] = np.array(temp)
    # 把 目标关键点矩阵 的维度，转置，变成2000 * 1
    b = t.T.reshape((2*num, 1))
    # 在两个矩阵上做线性回归，求直线的斜率和截距
    theta = np.linalg.lstsq(M, b)[0]
    # 最终的矩阵
    x = theta[:4].reshape((2, 2))
    y = theta[4:]
    return x, y
