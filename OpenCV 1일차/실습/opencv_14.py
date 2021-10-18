# OpenCV
# 이미지 중심에 반지름 10인 초록색 원을 그림
# 이미지 중심에 반지름이 100이고 선굵기가 1인 빨간색 원 그리기
# 원 아래에 “Draw Circle“ 글자 쓰기
import cv2
import numpy as np

height = 600
width = 800

center = (400, 300)
center_t = (250, 500)

radian_g = 10
radian_r = 100

color_g = (0, 255, 0)
color_r = (0, 0, 255)
color_t = (255, 255, 255)

img_zero = np.zeros((height, width, 3), dtype=np.uint8)

cv2.circle(img_zero, center, radian_g, color_g, thickness=-1)
cv2.circle(img_zero, center, radian_r, color_r, thickness=1)
cv2.putText(img_zero, "Draw Circle", center_t, cv2.FONT_HERSHEY_SIMPLEX, 2, color_t)

cv2.imshow('dst', img_zero)

cv2.waitKey(0)
cv2.destroyAllWindows()