# OpenCV 랜덤 원 출력
import cv2
import numpy as np
import random

height = 600
width = 800

img_zero = np.zeros((height, width, 3), dtype=np.uint8)

for i in range(20):
    center = (random.randint(50, 750), random.randint(50, 550))  # 그릴 원의 중심점(x,y)
    radian = random.randint(5, 15)   # 반지름
    color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) # RGB 칼라
    cv2.circle(img_zero, center, radian, color, thickness=random.randint(-1, 3))

cv2.imshow('dst', img_zero)
cv2.waitKey(0)
cv2.destroyAllWindows()