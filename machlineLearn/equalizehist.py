import cv2
src = cv2.imread("cat.png")
from skimage import io
# convert to gray
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# apply equalized hist
src_equ = cv2.equalizeHist(src)
cv2.imshow("cat_equ",src_equ)
io.imsave("cat_equ.png",src_equ)
cv2.waitKey()
