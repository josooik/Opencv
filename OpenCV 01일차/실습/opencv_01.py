# OpenCV 이미지, 그레이 출력
import cv2

img_src = cv2.imread('img/image.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

cont = cv2.hconcat([img_src, img_gray])

cv2.imshow('cont', cont)

cv2.waitKey(0)
cv2.destroyAllWindows()