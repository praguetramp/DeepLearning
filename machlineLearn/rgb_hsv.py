#Import Libraries
from skimage import io
from skimage import color
from matplotlib import pyplot as plt
from skimage import data
from pylab import *
#Read Image
img = io.imread("Samoyed.png")
#Convert to hsv
img_hsv = color.rgb2hsv(img)
#Convert back to rgb
img_rgb = color.hsv2rgb(img_hsv)
#Show both picture
figure (0)
io.imshow(img_hsv)
io.imsave("HSV_dog.jpg", img_hsv)
plt.show()
figure(1)
io.imshow(img_rgb)
io.imsave("RGB_dog.jpg", img_rgb)
plt.show()