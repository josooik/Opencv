# 흐림효과(blur)
# 컬러 -> 그레이스케일 -> 블러 -> 이진화
import cv2

img_src = cv2.imread('img/image.jpg', cv2.IMREAD_COLOR)
img_src = cv2.pyrDown(img_src)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)

filter = (7, 7)

# 흐림효과 적용
img_blur1 = cv2.blur(img_src, filter, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)
img_blur2 = cv2.blur(img_src, filter, anchor=(-1, -1), borderType=cv2.BORDER_REPLICATE)
img_blur3 = cv2.blur(img_src, filter, anchor=(-1, -1), borderType=cv2.BORDER_REFLECT)
img_blur4 = cv2.blur(img_src, filter, anchor=(-1, -1), borderType=cv2.BORDER_REFLECT101)
img_blur5 = cv2.blur(img_src, filter, anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
img_blur6 = cv2.blur(img_src, filter, anchor=(-1, -1), borderType=cv2.BORDER_ISOLATED)

cont = cv2.hconcat([img_src, img_grays, img_hsv, img_blur1, img_blur2])
cont1 = cv2.hconcat([img_blur3, img_blur4, img_blur5, img_blur6, img_blur6])

conts = cv2.vconcat([cont, cont1])

cv2.imshow('img', conts)
cv2.waitKey(0)
cv2.destroyAllWindows()