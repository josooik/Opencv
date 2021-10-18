# OpenCV 이미지 회전
import cv2

img_src = cv2.imread('img/img3.jpg', cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(500, 500), interpolation=cv2.INTER_AREA)
height, width = img_src.shape[:2]

center = (width / 2, height / 2)
angle = 45
scale = 1

# 회전
matrix = cv2.getRotationMatrix2D(center, angle, scale)
img_dst = cv2.warpAffine(img_src, matrix, (width, height))

cont = cv2.hconcat([img_src, img_dst])

cv2.imshow('img', cont)
cv2.waitKey(0)
cv2.destroyAllWindows()