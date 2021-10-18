# OpenCV 원 출력
import cv2
import numpy as np

height = 600
width = 800

center = (400, 300)
radian = 10
color = (0, 255, 0)

img_zero = np.zeros((height, width, 3), dtype=np.uint8)
cv2.circle(img_zero, center, radian, color, thickness=2)

cv2.imshow('dst', img_zero)

cv2.waitKey(0)
cv2.destroyAllWindows()