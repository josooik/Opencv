# OpenCV
# opencv_14 + opencv_15 + opencv_16 합치기
import cv2
import numpy as np

height = 600
width = 800

color_g = (0, 255, 0)
color_r = (0, 0, 255)
color_t = (255, 255, 255)
color_y = (0, 255, 255)

img_zero1 = np.zeros((height, width, 3), dtype=np.uint8)
img_zero2 = np.zeros((height, width, 3), dtype=np.uint8)
img_zero3 = np.zeros((height, width, 3), dtype=np.uint8)

cv2.circle(img_zero1, (400, 300), 10, color_g, thickness=-1)
cv2.circle(img_zero1, (400, 300), 100, color_r, thickness=1)
cv2.putText(img_zero1, "Draw Circle", (250, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, color_t)

cv2.line(img_zero2, (0, 0), (800, 600), color_g, thickness=3)
cv2.line(img_zero2, (800, 0), (0, 600), color_r, thickness=3)
cv2.putText(img_zero2, "Draw line", (250, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, color_t)

cv2.ellipse(img_zero3, (400, 300), (250, 250), 0, 0, 360, color_r, 2)
cv2.ellipse(img_zero3, (400, 300), (250, 10), 0, 0, 360, color_y, 2)
cv2.ellipse(img_zero3, (400, 300), (10, 250), 0, 0, 360, color_g, 2)
cv2.putText(img_zero3, "Draw EllipseCircle", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color_t)

cont1 = cv2.hconcat([img_zero1, img_zero2])
cont2 = cv2.hconcat([cont1, img_zero3])

cv2.imshow('img', cont2)

cv2.waitKey(0)
cv2.destroyAllWindows()