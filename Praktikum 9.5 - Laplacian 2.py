import cv2

#load citra
img = cv2.imread('PicsArt_09-12-09.03.11.jpg',0)

#aplikasikan gaussian
blur = cv2.GaussianBlur(img,(3,3),0)

#aplikasikan laplacian
laplacian = cv2.Laplacian(blur,cv2.CV_64F)
laplacian = laplacian/laplacian.max()

cv2.imshow('laplacian-gaussian',laplacian)
cv2.waitKey(0)