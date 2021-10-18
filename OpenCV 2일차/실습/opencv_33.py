# OpenCV 이미지 반전
import cv2

img_src = cv2.imread('img/img2.png', cv2.IMREAD_COLOR)

img_dst = cv2.bitwise_not(img_src)

cont = cv2.hconcat([img_src, img_dst])

cv2.imshow('img', cont)
cv2.waitKey(0)
cv2.destroyAllWindows()