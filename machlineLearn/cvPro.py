import cv2
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt

from skimage import *

#Read pic1
img1 = cv2.imread("Samoyed.png")
#Read pic2
img2 = cv2.imread("cat.png")
#resize cat  to the same
img2_res = resize(img2, (198, 308))
io.imsave("cat_res.png", img2_res)
img3 =io.imread("cat_res.png")
#Define alpha and beta
alpha =0.30
beta = 0.70
#Blend Images
final_image = cv2.addWeighted(img1, alpha, img3, beta, 0.0)

#show
io.imshow(final_image)
io.imsave("blend_cat_dog.png",final_image)
plt.show()
