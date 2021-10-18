# OpenCV 랜덤 사각형 출력
import cv2
import numpy as np
import random

height = 600
width = 800

img_zero = np.zeros((height, width, 3), dtype=np.uint8)

for i, n in enumerate(range(50, 600, 50)):
    cv2.rectangle(img_zero, (50, 50), (230+50 * i, 100 + 50 * i), (random.randint(0, 1) * 200 + 55, random.randint(0, 1) * 200 + 55, random.randint(0, 1) * 200 + 55), thickness=2)

cv2.imshow('dst', img_zero)
cv2.waitKey(0)
cv2.destroyAllWindows()