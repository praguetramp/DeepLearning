# Import package
import cv2
from skimage import io
from matplotlib import pyplot as plt

img = cv2.imread("Samoyed.png")
# 二值化：大于阈值为255（白）小于为黑色(0) [0,1]
# 反向二值化：大于为0，小于为1 [0,1]
# 截断：大于为阈值，小于不变 [0,x]
# 阈限到零：大于不变，小于为0   0+[x,255]
# 反向阈限到零：大于为0，小于不变 [0,x]

# different types
# 返回值有两个参数，第二个参数才是图像
_, img1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
_, img2 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
_, img3 = cv2.threshold(img, 50, 255, cv2.THRESH_TRUNC)
_, img4 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
_, img5 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO_INV)
# Show different images

#阈值二值化
io.imshow(img1)
io.imsave("bin_d.png", img1)
plt.show()
#阈值反向二值化
io.imshow(img2)
io.imsave("bin_inv_d.png", img2)
plt.show()
#截断
io.imshow(img3)
io.imsave("trunc_d.png", img3)
plt.show()
#阈限到零
io.imshow(img4)
io.imsave("thr_o_d.png", img4)
plt.show()
#阈限反向到零
io.imshow(img5)
io.imsave("thr_o_inv_d.png", img5)
plt.show()
