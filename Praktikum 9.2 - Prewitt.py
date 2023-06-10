#Import library
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Read input image from file
img = cv.imread("E:/PicsArt_09-12-09.03.11.jpg", cv.IMREAD_GRAYSCALE)

#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

img_prewittx = cv.filter2D(img, -1, kernelx)
img_prewitty = cv.filter2D(img, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

#Plot output image
fig, axes = plt.subplots(4, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins=256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_prewittx, cmap='gray')
ax[2].set_title("Citra Output Prewitt X")
ax[3].hist(img_prewittx.ravel(), bins=256)
ax[3].set_title("Histogram Citra Output Prewitt X")

ax[4].imshow(img_prewitty, cmap='gray')
ax[4].set_title("Citra Output Prewitt Y")
ax[5].hist(img_prewitty.ravel(), bins=256)
ax[5].set_title("Histogram Citra Output Prewitt Y")

ax[6].imshow(img_prewitt, cmap='gray')
ax[6].set_title("Citra Output Prewitt")
ax[7].hist(img_prewitt.ravel(), bins=256)
ax[7].set_title("Histogram Citra Output Prewitt")

fig.tight_layout()
plt.show()