# OpenCV 사다리꼴 영역 출력
import cv2
import numpy as np

trap_bottom_width = 0.8
trap_top_width = 0.1
trap_height = 0.4

img_bgr = cv2.imread('img/img19.png', cv2.IMREAD_COLOR)
img_src = np.zeros_like(img_bgr)
height, width = img_src.shape[:2]

pts = np.array([[
    ((width * (1 - trap_bottom_width)) // 2, height),
    ((width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
    (width - (width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
    (width - (width * (1 - trap_bottom_width)) // 2, height)]],
    dtype=np.int32)

src = cv2.fillPoly(img_src, pts, (255, 255, 255), cv2.LINE_AA)

cv2.imshow("src", src)
cv2.waitKey()
cv2.destroyAllWindows()