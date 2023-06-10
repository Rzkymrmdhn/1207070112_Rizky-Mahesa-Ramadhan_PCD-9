import cv2
from matplotlib import pyplot as plt

#baca gambar
img0 = cv2.imread('PicsArt_09-12-09.03.11.jpg')

#konversi ke grays
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

#hilangkan noise
img = cv2.GaussianBlur(gray,(3,3),0)

#konvolusi dengan kernel
laplacian = cv2.Laplacian(img,cv2.CV_64F)

#tampilkan dengan matplotlib
plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()