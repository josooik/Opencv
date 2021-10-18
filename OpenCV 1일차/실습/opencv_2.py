# OpenCV 이미지, 그레이, 합성이미지 출력
import cv2

img_src = cv2.imread('img/image.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

dst = img_src[100:430, 100:460].copy()
width, height, cannel = dst.shape

cont = cv2.hconcat([img_src, img_grays])

img_grays1 = img_grays[100:430, 100:460]
img_grays1[0:width, 0:height] = dst

cont1 = cv2.hconcat([cont, img_grays])

cv2.imshow('img', cont1)

cv2.waitKey(0)
cv2.destroyAllWindows()