from skimage import exposure
from matplotlib import pyplot as plt
from skimage import io
from pylab import *
img = io.imread("Samoyed.png")

# gamma correct
gamma_correct1 = exposure.adjust_gamma(img, 0.5)
gamma_correct2 = exposure.adjust_gamma(img, 5)
figure(0)
io.imshow(gamma_correct1)
plt.show()
io.imsave("gamma_c1.jpg", gamma_correct1)
figure(1)
io.imshow(gamma_correct2)
plt.show()
io.imsave("gamma_c2.jpg", gamma_correct2)
