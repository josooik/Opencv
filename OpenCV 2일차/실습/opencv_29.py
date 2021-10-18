# OpenCV 이미지 대칭
import cv2

img_src = cv2.imread('img/img3.jpg', cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(500, 500), interpolation=cv2.INTER_AREA)
img_flip_lr = cv2.flip(img_src, 1)  # 1 : 좌우대칭 / 0 : 상하대칭 / -1 : 상하좌우 대칭
img_flip_ud = cv2.flip(img_src, 0)
img_flip_lrud = cv2.flip(img_src, -1)

cont = cv2.hconcat([img_src, img_flip_lr])
cont1 = cv2.hconcat([img_flip_ud, img_flip_lrud])
cont2 = cv2.vconcat([cont, cont1])

cv2.imshow('img', cont2)
cv2.waitKey(0)
cv2.destroyAllWindows()