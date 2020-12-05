import numpy as np
import cv2

def nothing(x):
	pass

img = cv2.imread('khe.jpg')
rows, cols, ch = img.shape
mask = np.zeros(img.shape[:2], np.uint8)

cv2.namedWindow("Argument", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)

cv2.createTrackbar("a", "Argument", 0, 100, nothing)
cv2.createTrackbar("b", "Argument", 0, 100, nothing)
cv2.createTrackbar("c", "Argument", 0, cols, nothing)
cv2.createTrackbar("d", "Argument", 0, rows, nothing)
Switch = "0: OFF \n1: ON"
cv2.createTrackbar(Switch, "Argument", 0, 1, nothing)

while(1):
	im = img

	a = cv2.getTrackbarPos('a', 'Argument')
	b = cv2.getTrackbarPos('b', 'Argument')
	c = cv2.getTrackbarPos('c', 'Argument')
	if c == 0:
		c = 1
	d = cv2.getTrackbarPos('d', 'Argument')
	if d == 0:
		d = 1
	s = cv2.getTrackbarPos(Switch, "Argument")


	if s == 1:
		bgdModel = np.zeros((1,65), np.float64)
		fgdModel = np.zeros((1,65), np.float64)

		rect = (a, b, c, d)
		cv2.grabCut(im, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
		mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
		im = im*mask2[:,:,np.newaxis]

	if s == 0:
		im = img

	cv2.imshow("result", im)
	
	k = cv2.waitKey(1) & 0xFF
	if k == 27: 
		break

cv2.destroyAllWindows()