# OpenCV 이미지 여러색 출력
import cv2
import numpy as np

img_src = cv2.imread('img/image.jpg', cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(500, 500), interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

img_b, img_g, img_r = cv2.split(img_src)  # 채널 분리
height, width = img_src.shape[:2]

zero = np.zeros((height, width, 1), dtype=np.uint8)
zeros = cv2.merge((zero, zero, zero))

img_bs = cv2.merge((img_b, zero, zero))
img_gs = cv2.merge((zero, img_g, zero))
img_rs = cv2.merge((zero, zero, img_r))

cont = cv2.hconcat([img_src, img_grays])
cont1 = cv2.hconcat([cont, zeros])

cont2 = cv2.hconcat([img_bs, img_gs])
cont3 = cv2.hconcat([cont2, img_rs])

cont4 = cv2.vconcat([cont1, cont3])

cv2.imshow('img', cont4)

cv2.waitKey(0)
cv2.destroyAllWindows()