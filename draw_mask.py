import cv2
import numpy as np


mask = cv2.imread("mask.png")
mask = cv2.resize(mask, (640,480))
img = cv2.imread("test_image.jpg")
#img = cv2.resize(img, (720,1280))
mask[mask > 0] = 255
inv_mask = cv2.bitwise_not(mask)
blank = np.zeros((480, 640, 3), np.uint8)
blank[:] = (255, 0, 0)
apples = cv2.bitwise_and(blank, mask)

background = cv2.bitwise_and(img, inv_mask)

final = cv2.bitwise_or(background, apples)
cv2.imwrite("res.jpg", final)
cv2.imshow("sd", final)

cv2.waitKey(0)
cv2.destroyAllWindows()