# OpenCV 이미지 확대 축소 크기조절
import cv2

img_src = cv2.imread('img/img3.jpg', cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(500, 500), interpolation=cv2.INTER_AREA)
height, width = img_src.shape[:2]

# 확대
img_UP = cv2.pyrUp(img_src, dstsize=(width * 2, height * 2), borderType=cv2.BORDER_DEFAULT)

# 축소
img_DOWN = cv2.pyrDown(img_src)

# 크기조절 1
img_resize1 = cv2.resize(img_src, dsize=(int(width * 0.7), int(height * 0.7)), interpolation=cv2.INTER_LINEAR)

# 크기조절 2
img_resize2 = cv2.resize(img_src, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)

cv2.imshow('img_UP', img_UP)
cv2.imshow('img_DOWN', img_DOWN)
cv2.imshow('img_resize1', img_resize1)
cv2.imshow('img_resize2', img_resize2)

cv2.waitKey(0)
cv2.destroyAllWindows()