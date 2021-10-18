# OpenCV
# 타원의 중심  : 영상의 가운데
# x축 방향으로 길이 200 y축 방향으로 길이 200인 빨간색 타원 그리기
# x축 방향으로 길이 200 y축 방향으로 길이 10인 노란색 타원 그리기
# x축 방향으로 길이 10 y축 방향으로 길이 200인 녹색색 타원 그리기
# 빨간 타원 안쪽에 “Draw EllipseCircle“
import cv2
import numpy as np

height = 600
width = 800

ellipse1_start = (400, 300)
ellipse1_end = (250, 250)

ellipse2_start = (400, 300)
ellipse2_end = (250, 10)

ellipse3_start = (400, 300)
ellipse3_end = (10, 250)

color_g = (0, 255, 0)
color_r = (0, 0, 255)
color_y = (0, 255, 255)
color_t = (255, 255, 255)

center_t = (250, 450)

img_zero = np.zeros((height, width, 3), dtype=np.uint8)

cv2.ellipse(img_zero, ellipse1_start, ellipse1_end, 0, 0, 360, color_r, 2)
cv2.ellipse(img_zero, ellipse2_start, ellipse2_end, 0, 0, 360, color_y, 2)
cv2.ellipse(img_zero, ellipse3_start, ellipse3_end, 0, 0, 360, color_g, 2)

cv2.putText(img_zero, "Draw EllipseCircle", center_t, cv2.FONT_HERSHEY_SIMPLEX, 1, color_t)

cv2.imshow('dst', img_zero)

cv2.waitKey(0)
cv2.destroyAllWindows()