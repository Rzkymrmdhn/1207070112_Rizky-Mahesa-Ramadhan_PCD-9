#Import library
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Read input image from file
img = cv.imread("E:/PicsArt_09-12-09.03.11.jpg", cv.IMREAD_GRAYSCALE)

#Proses sobel
img_sobelx = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)
img_sobely = cv.Sobel(img, cv.CV_8U, 0, 1, ksize=5)
img_sobel = img_sobelx + img_sobely

#Plot output image
fig, axes = plt.subplots(4, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins=256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_sobelx, cmap='gray')
ax[2].set_title("Citra Output Sobel X")
ax[3].hist(img_sobelx.ravel(), bins=256)
ax[3].set_title("Histogram Citra Output Sobel X")

ax[4].imshow(img_sobely, cmap='gray')
ax[4].set_title("Citra Output Sobel Y")
ax[5].hist(img_sobely.ravel(), bins=256)
ax[5].set_title("Histogram Citra Output Sobel Y")

ax[6].imshow(img_sobel, cmap='gray')
ax[6].set_title("Citra Output Sobel")
ax[7].hist(img_sobel.ravel(), bins=256)
ax[7].set_title("Histogram Citra Output Sobel")

fig.tight_layout()
plt.show()