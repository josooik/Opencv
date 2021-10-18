# opencv_47 + opencv_48
# 모폴로지 연산
# 형태소(Structing element)의 크기와 형태를 지정
# 크기 : 3x3 5x5 7x7 9x9
# 형태 : 십자가모양(cv2.MORPH_CROSS), 직사각형(cv2.MORPH_RECT), 타원(cv2.MORPH_ELLIPSE)
# kernel = cv2.getStucturingElement(형태, 형태소크기(9,9)
import cv2
import numpy as np

img_src1 = cv2.imread('img/img16.png', cv2.IMREAD_COLOR)
img_src2 = cv2.imread('img/img16.png', cv2.IMREAD_COLOR)

img_gray = cv2.cvtColor(img_src1, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

my_color = (0, 255, 0)
text_color = (255, 0, 0)
thickness = 2

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

ret, img_binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
mask = cv2.merge((img_binary, img_binary, img_binary))

# CLOSING 5번은 dilate5번 진행후 erode5번 진행하는것과 같다.
img_morp1 = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=5)
mask_morp1 = cv2.merge((img_morp1, img_morp1, img_morp1))

# OPENING 5번은 erode5번 진행후 dilate5번 진행하는것과 같다.
img_morp2 = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=5)
mask_morp2 = cv2.merge((img_morp2, img_morp2, img_morp2))

#contours, hierachy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours, hierachy = cv2.findContours(img_morp1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#contours, hierachy = cv2.findContours(img_morp2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)

    if area > 1000:
        cv2.drawContours(img_src2, contours, i, my_color, thickness)

        mu = cv2.moments(contour)
        cx = int(mu['m10'] / (mu['m00'] + 1e-5))
        cy = int(mu['m01'] / (mu['m00'] + 1e-5))

        cv2.circle(img_src2, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(img_src2, f'{i}: {int(area)}', (cx-50, cy-20), cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 1)

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_src2, (x, y), (x+w, y+h), (255, 255, 0), 1)

        cont = cv2.hconcat([img_src1, img_grays, img_src2])
        cont1 = cv2.hconcat([mask, mask_morp1, mask_morp2])

        conts = cv2.vconcat([cont, cont1])

        cv2.imshow('img', conts)

cv2.waitKey(0)

cv2.destroyAllWindows()