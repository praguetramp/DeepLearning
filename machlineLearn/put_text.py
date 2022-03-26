import cv2
import numpy as py
from skimage import io
from matplotlib import pyplot as plt

#Read Image
img = cv2.imread("cat_res.png")
#Set Font
font = cv2.FONT_HERSHEY_SIMPLEX
#Write on the image
cv2.putText(img,"I am a cat of csy !",(20,100),font,0.8,(0,255,0),2,cv2.LINE_4)
io.imshow(img)
io.imsave("text_cat.png", img)
plt.show()