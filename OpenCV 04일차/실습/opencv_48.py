# 모폴로지 연산
# 형태소(Structing element)의 크기와 형태를 지정
# 크기 : 3x3 5x5 7x7 9x9
# 형태 : 십자가모양(cv2.MORPH_CROSS), 직사각형(cv2.MORPH_RECT), 타원(cv2.MORPH_ELLIPSE)
# kernel = cv2.getStucturingElement(형태, 형태소크기(9,9)
import cv2
import numpy as np

img_src = cv2.imread('img/img17.png', cv2.IMREAD_COLOR)
img_src = cv2.pyrDown(img_src)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

img_gray = cv2.bitwise_not(img_gray)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Erosion 은 이미지를 침식시키는 것
# Foreground 가 되는 이미지의 경계부분을 침식시켜서 Background 이미지로 전환한다.
# Foreground 이미지가 가늘게 된다.
# 흐릿한 경계부분은 배경으로 만들어버린다고 생각하면 좋을 듯
# 이미지는 Matrix 이다. 작은 (n x n) 커널 창으로 이미지 전체를 훝으면서 커널창에 들어온 matrix 값들을 변경한다.
# erosion = cv2.erode(img, kernel, iterations=1)
# img: erosion을 수행할 원본 이미지
# kernel: erosion을 위한 커널
# iterations: Erosion 반복 횟수
img_erode = cv2.erode(img_gray, kernel, iterations=5)
mask_erode = cv2.merge((img_erode, img_erode, img_erode))

# Dilation 은 이미지를 팽창시키는 것
# Erosion과 반대로 동작한다.
# dilation = cv2.dilate(img, kernel, iterations=1)
# img: erosion을 수행할 원본 이미지
# kernel: erosion을 위한 커널
# iterations: Erosion 반복 횟수
img_dilate = cv2.dilate(img_gray, kernel, iterations=5)
mask_dilate = cv2.merge((img_dilate, img_dilate, img_dilate))

img_morp = img_gray.copy()

img_morp = cv2.erode(img_morp, kernel, iterations=5)
img_morp = cv2.dilate(img_morp, kernel, iterations=5)
'''
img_morp = cv2.dilate(img_morp, kernel, iterations=3)
img_morp = cv2.erode(img_morp, kernel, iterations=3)
'''
mask_morp = cv2.merge((img_morp, img_morp, img_morp))

img_morps = img_gray.copy()

# CLOSING 5번은 dilate5번 진행후 erode5번 진행하는것과 같다.
img_morp1 = cv2.morphologyEx(img_morps, cv2.MORPH_CLOSE, kernel, iterations=5)
mask_morp1 = cv2.merge((img_morp1, img_morp1, img_morp1))

# OPENING 5번은 erode5번 진행후 dilate5번 진행하는것과 같다.
img_morp2 = cv2.morphologyEx(img_morps, cv2.MORPH_OPEN, kernel, iterations=5)
mask_morp2 = cv2.merge((img_morp2, img_morp2, img_morp2))

cont = cv2.hconcat([img_src, img_grays, mask_erode, mask_morp1])
cont1 = cv2.hconcat([img_src, mask_morp, mask_dilate, mask_morp2])

conts = cv2.vconcat([cont, cont1])

cv2.imshow('img', conts)

cv2.waitKey(0)
cv2.destroyAllWindows()