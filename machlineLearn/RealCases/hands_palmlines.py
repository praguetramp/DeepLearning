import cv2

img = cv2.imread('palm.png')
cv2.imshow("original hands:", img)
cv2.waitKey()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 用过滤算法canny边缘检测器找出掌纹
edges = cv2.Canny(gray, 85, 120, apertureSize=3)
cv2.imshow("edges in palm :", edges)
cv2.waitKey()
# 反转颜色，使得识别出来的掌纹线时黑色的
edges = cv2.bitwise_not(edges)
cv2.imshow("revert in edges :", edges)
cv2.waitKey()

cv2.imwrite('palmlines.png', edges)
palmlines = cv2.imread('palmlines.png')
img = cv2.addWeighted(palmlines, 0.3, img, 0.7, 0)
cv2.imshow("final palmlines:", img)
cv2.waitKey()
