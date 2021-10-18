# OpenCV 다각형 출력
import cv2
import numpy as np

height = 600
width = 800

img_zero = np.zeros((height, width, 3), dtype=np.uint8)

points = np.array([[300, 400], [170, 10], [200, 230], [70, 70], [50, 150]], np.int32)
points1 = np.array([[350,250], [450,350], [400,450], [300,450], [250,350]], np.int32)

cv2.polylines(img_zero, [points], False, (0, 255, 0), 2)
cv2.polylines(img_zero, [points1], True, (205, 255, 0), 5)

cv2.imshow('dst', img_zero)
cv2.waitKey(0)
cv2.destroyAllWindows()