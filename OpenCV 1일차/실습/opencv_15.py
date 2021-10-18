# OpenCV
# 선굵기는 3입니다.
# 대각선 아래에 “Draw Line“ 글자 쓰기
import cv2
import numpy as np

height = 600
width = 800

line1_start = (0, 0)
line1_end = (800, 600)

line2_start = (800, 0)
line2_end = (0, 600)

line1_color_g = (0, 255, 0)
line2_color_r = (0, 0, 255)
color_t = (255, 255, 255)

center_t = (250, 500)

img_zero = np.zeros((height, width, 3), dtype=np.uint8)

cv2.line(img_zero, line1_start, line1_end, line1_color_g, thickness=3)
cv2.line(img_zero, line2_start, line2_end, line2_color_r, thickness=3)

cv2.putText(img_zero, "Draw line", center_t, cv2.FONT_HERSHEY_SIMPLEX, 2, color_t)

cv2.imshow('dst', img_zero)

cv2.waitKey(0)
cv2.destroyAllWindows()