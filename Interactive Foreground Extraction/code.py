import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('dep.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (10, 10, 375, 500)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
# cv2.imwrite('newmask.png', img)
# cv2.imshow('img', img)

# newmask is the mask image I manually labelled
newmask = cv2.imread("newmask.png", 0)
img1 = cv2.imread('dep.jpg')
# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
cv2.imshow("mask", mask)
mask, bgdModel, fgdModel = cv2.grabCut(img1, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

mask = np.where((mask==2)|(mask==0),0,1).astype("uint8")
img1 = img1*mask[:,:,np.newaxis]

# cv2.imshow('img1', img1)
cv2.waitKey()
cv2.destroyAllWindows()

# plt.imshow(img),plt.colorbar(),plt.show()

# plt.imshow(img),plt.colorbar(),plt.show()

# cv2.imshow("kết quả", img)
# cv2.waitKey()
# cv2.destroyAllWindows()