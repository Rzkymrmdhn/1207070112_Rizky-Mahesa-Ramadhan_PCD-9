#Import library
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Read input image from file
img = cv.imread("E:/PicsArt_09-12-09.03.11.jpg", cv.IMREAD_GRAYSCALE)

#Canny
img_canny = cv.Canny(img,100,200)

#Plot output image
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins=256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_canny, cmap='gray')
ax[2].set_title("Citra Output")
ax[3].hist(img_canny.ravel(), bins=256)
ax[3].set_title("Histogram Citra Output")

fig.tight_layout()
plt.show()