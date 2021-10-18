# OpenCV HSV 채널 나누기
import cv2
import numpy as np

img_src = cv2.imread('img/img10.jpg', cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(950, 400), interpolation=cv2.INTER_AREA)

img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
img_h, img_s, img_v = cv2.split(img_hsv)

height, width = img_src.shape[:2]

zero = np.zeros((height, width, 1), dtype=np.uint8)
zeros = cv2.merge((zero, zero, zero))

img_hs = cv2.merge((img_h, img_h, img_h))
img_ss = cv2.merge((img_s, img_s, img_s))
img_vs = cv2.merge((img_v, img_v, img_v))

cont = cv2.hconcat([img_src, img_hs])
cont1 = cv2.hconcat([img_ss, img_vs])
cont2 = cv2.vconcat([cont, cont1])

print(cont1.shape)
cv2.imshow("img", cont2)
cv2.waitKey(0)
cv2.destroyAllWindows()