# OpenCV 이미지 자르기
import cv2

img_src = cv2.imread('img/img11.jpg', cv2.IMREAD_COLOR)

img_dst = img_src[245:830, 45:310, :].copy()
img_dst1 = img_src[440:790, 352:624, :].copy()
img_dst2 = img_src[768:792, 972:450, :].copy()
'''

img_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
img_dst = cv2.cvtColor(img_dst, cv2.COLOR_GRAY2BGR)
img_src[245:830, 45:310, :] = img_dst

img_dst1 = cv2.cvtColor(img_dst1, cv2.COLOR_BGR2GRAY)
img_dst1 = cv2.cvtColor(img_dst1, cv2.COLOR_GRAY2BGR)
img_src[353:441, 788:622, :] = img_dst1

img_dst2 = cv2.cvtColor(img_dst2, cv2.COLOR_BGR2GRAY)
img_dst2 = cv2.cvtColor(img_dst2, cv2.COLOR_GRAY2BGR)
img_src[768:792, 972:450, :] = img_dst2
'''

cv2.imshow('img_dst', img_dst)
cv2.imshow('img_dst1', img_dst1)
#cv2.imshow('img_dst2', img_dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()