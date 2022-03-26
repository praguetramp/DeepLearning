import skimage
from skimage import io
from matplotlib import pyplot as plt
import pandas as pd
img = io.imread("Samoyed.png")
io.imshow(img)
#plt.show()
#Getting Image Resolution
print(img.shape)
df = pd.DataFrame(img.flatten())
filepath = 'pixel_values1.xlsx'
print(df)
df.to_excel(filepath)