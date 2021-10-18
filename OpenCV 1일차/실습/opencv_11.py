# OpenCV 타원 출력
import cv2
import numpy as np

height = 600
width = 800

img_zero = np.zeros((height, width, 3), dtype=np.uint8)

cv2.ellipse(img_zero, (250, 400), (100, 50), 0, 0, 360, (0, 255, 0), 2)
cv2.ellipse(img_zero, (650, 400), (50, 100), 0, 0, 360, (255, 255, 0), 2)
cv2.ellipse(img_zero, (550, 200), (50, 100), 20, 50, 360, (255, 255, 255), -1)

cv2.imshow('dst', img_zero)
cv2.waitKey(0)
cv2.destroyAllWindows()