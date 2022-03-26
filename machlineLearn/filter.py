import cv2
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
#Read image for different blurring
img_origental = cv2.imread("cat.png")
img_medianBlur = cv2.imread("cat.png")
img_gaussianBlur = cv2.imread("cat.png")
img_bilateralBlur = cv2.imread("cat.png")

#Blur images
#中值滤波，模糊处理的同时，保留边  依赖临近像素的中值(中位数)
img_medianBlur = cv2.medianBlur(img_medianBlur, 9)
#高斯滤波，模糊处理，不考虑边   依赖图像整体的标准差
img_gaussianBlur = cv2.GaussianBlur(img_gaussianBlur,(9,9),10)
#双边滤波，在平滑处理的同时保留边的完整性  依赖与临近像素的均值
img_bilateralBlur = cv2.bilateralFilter(img_bilateralBlur,9,100,75)

#Show Imag
io.imshow(img_origental)

plt.show()
io.imshow(img_medianBlur)
io.imsave("medianBlur_cat.png", img_medianBlur)
plt.show()
io.imshow(img_gaussianBlur)
io.imsave("gaussianBlur_cat.png", img_gaussianBlur)
plt.show()
io.imshow(img_bilateralBlur)
io.imsave("bilateralBlur_cat.png", img_bilateralBlur)
plt.show()