# 이진화
import cv2

img_src = cv2.imread('img/img14.png', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

img_src1 = cv2.imread('img/img15.png', cv2.IMREAD_COLOR)
img_gray1 = cv2.cvtColor(img_src1, cv2.COLOR_BGR2GRAY)
img_grays1 = cv2.cvtColor(img_gray1, cv2.COLOR_GRAY2BGR)


# 이진화를 진행
# 이미지의 특성을 파악 : 검출할려고 하는것(도형)이 흰색으로 나와야함
# 배경이 흰색 : 검출해야하는 물체보다 배경이 밝은 상태
# 방법 2-1 : 그레이이미지를 반전하고 Threshold를 적용

gray1 = cv2.bitwise_not(img_gray)
ret, img_binary = cv2.threshold(gray1, 50, 255, cv2.THRESH_BINARY)
ret1, img_binary1 = cv2.threshold(gray1, 80, 255, cv2.THRESH_BINARY)

ret2, img_binary2 = cv2.threshold(img_gray1, 50, 255, cv2.THRESH_BINARY)
ret3, img_binary3 = cv2.threshold(img_gray1, 80, 255, cv2.THRESH_BINARY)

mask = cv2.merge((img_binary, img_binary, img_binary))
mask1 = cv2.merge((img_binary1, img_binary1, img_binary1))

mask2 = cv2.merge((img_binary2, img_binary2, img_binary2))
mask3 = cv2.merge((img_binary3, img_binary3, img_binary3))

cont = cv2.hconcat([img_src, img_grays, mask, mask1])
cont1 = cv2.hconcat([img_src1, img_grays1, mask2, mask3])

conts = cv2.vconcat([cont, cont1])

cv2.imshow('img', conts)

cv2.waitKey(0)
cv2.destroyAllWindows()