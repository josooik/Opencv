# OpenCV 랜덤 선 출력
import cv2
import numpy as np
import random

height = 600
width = 800

img_zero = np.zeros((height, width, 3), dtype=np.uint8)

for i, n in enumerate(range(50, 600, 50)):
    cv2.line(img_zero, (50, n), (750, n), (random.randint(0, 1) * 200 + 55, random.randint(0, 1) * 200 + 55, random.randint(0, 1) * 200 + 55), thickness= i + 1)

cv2.imshow('dst', img_zero)
cv2.waitKey(0)
cv2.destroyAllWindows()