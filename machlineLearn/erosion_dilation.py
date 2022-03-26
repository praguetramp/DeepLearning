#侵蚀：维护一个领域核，用领域最小值代替该像素值，使物体边界的像素值减少
#扩张：维护一个领域核，用最大值代替，使得物体边界的像素值增大
import cv2
from skimage import io
from matplotlib import pyplot as plt
img = cv2.imread("cat.png")
#Define erosion size
s1 = 0
s2 = 10
s3 = 10
#Define erosion type
#矩形核
t1 = cv2.MORPH_RECT
#十字形核
t2 = cv2.MORPH_CROSS
#椭圆形核
t3 = cv2.MORPH_ELLIPSE
#Define and save the errsion template
#定义核函数：侵蚀、扩张的类型，核的大小，核的起始点
tmp1 = cv2.getStructuringElement(t1,(2*s1+1,2*s1+1),(s1,s1))
tmp2 = cv2.getStructuringElement(t2,(2*s2+1,2*s2+1),(s2,s2))
tmp3 = cv2.getStructuringElement(t3,(2*s3+1,2*s3+1),(s3,s3))
#Apply the erosion template to the image  and save in different variables
final1 = cv2.erode(img,tmp1)
final2 = cv2.erode(img,tmp2)
final3 = cv2.erode(img,tmp3)
#show all the images with different erosions
io.imshow(final1)
io.imsave("ero1_cat.png",final1)
plt.show()

io.imshow(final2)
io.imsave("ero2_cat.png",final2)
plt.show()

io.imshow(final3)
io.imsave("ero3_cat.png",final3)
plt.show()

#Dilation code:

#Define dilation
d1 = 0
d2 = 10
d3 = 20
#Store the dilation template
tmp4 = cv2.getStructuringElement(t1,(2*d1+1,2*d1+1),(d1,d1))
tmp5 = cv2.getStructuringElement(t2,(2*d2+1,2*d2+1),(d2,d2))
tmp6 = cv2.getStructuringElement(t3,(2*d3+1,2*d3+1),(d3,d3))
#Apply dilation to the images
final4 = cv2.dilate(img,tmp4)
final5 = cv2.dilate(img,tmp5)
final6 = cv2.dilate(img,tmp6)
#Show the images
io.imshow(final4)
io.imsave("dil1_cat.png",final4)
plt.show()
io.imshow(final5)
io.imsave("dil2_cat.png",final5)
plt.show()
io.imshow(final5)
io.imsave("dil3_cat.png",final6)
plt.show()