# OpenCV 이미지 채널별로 출력
import cv2

img_src = cv2.imread('img/image.jpg', cv2.IMREAD_COLOR)

img_b, img_g, img_r = cv2.split(img_src)  # 채널 분리
img_bgr = cv2.merge((img_b, img_g, img_r)) # 채널 합치기

cv2.imshow('img', img_src)

cv2.imshow('img_b', img_b)
cv2.imshow('img_g', img_g)
cv2.imshow('img_r', img_r)

cv2.imshow('img_bgr', img_bgr)
ㄴ
cv2.waitKey(0)
cv2.destroyAllWindows()