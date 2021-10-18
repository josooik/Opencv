# 이진화
import cv2

img_src = cv2.imread('img/img13.jpg', cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(300, 300), interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)

# 이진화
# THRESH_BINARY : img_dst = (img_src > thr) ? max_value : 0
_, img_binary1 = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)
# THRESH_BINARY_INV : img_dst = (img_src > thr) ? 0 : max_value
_, img_binary2 = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY_INV)
# THRESH_TRUNC : img_dst = (img_src > thr) ? thr : img_src
_, img_binary3 = cv2.threshold(img_gray, 160, 255, cv2.THRESH_TRUNC)
# THRESH_TOZERO : img_dst = (img_src > thr) ? img_src : 0
_, img_binary4 = cv2.threshold(img_gray, 160, 255, cv2.THRESH_TOZERO)
# THRESH_TOZERO_INV : img_dst = (img_src > thr) ? 0 : img_src
_, img_binary5 = cv2.threshold(img_gray, 160, 255, cv2.THRESH_TOZERO_INV)
_, img_binary6 = cv2.threshold(img_gray, 160, 255, cv2.THRESH_OTSU)

mask1 = cv2.merge((img_binary1, img_binary1, img_binary1))
mask2 = cv2.merge((img_binary2, img_binary2, img_binary2))
mask3 = cv2.merge((img_binary3, img_binary3, img_binary3))
mask4 = cv2.merge((img_binary4, img_binary4, img_binary4))
mask5 = cv2.merge((img_binary5, img_binary5, img_binary5))
mask6 = cv2.merge((img_binary6, img_binary6, img_binary6))

cont = cv2.hconcat([img_src, img_grays, img_hsv])
bin1 = cv2.hconcat([mask1, mask2, mask3])
bin2 = cv2.hconcat([mask4, mask5, mask6])
conts = cv2.vconcat([cont, bin1, bin2])

cv2.imshow('img', conts)
cv2.waitKey(0)
cv2.destroyAllWindows()