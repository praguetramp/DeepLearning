import skimage
import cv2
from skimage import io
from pylab import *
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread("Samoyed.png")

#Create a dummy image that stores different contrast and brightness
new_image = np.zeros(image.shape, image.dtype)

#brightness and contrast parameters

contrast = 3.0
brightness = 2.0

#Change the contrast and brightness
for y in range (image.shape[0]) :
    for x in range (image.shape[1]) :
        for c in range (image.shape[2]):
            new_image[y,x,c] = np.clip(contrast *image[y,x,c]+brightness, 0, 255)
#old pic
cv2.imshow("old_dog",image)
cv2.waitKey()
#new pic

cv2.imshow("new_dog",new_image)
io.imsave("con_bri_changed_dog.png",new_image)
cv2.waitKey()