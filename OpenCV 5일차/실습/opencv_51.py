# pcb 부품 인식
import cv2
import numpy as np

img_src = cv2.imread('img/PCB1/img1.bmp', cv2.IMREAD_COLOR)
#img_src = cv2.resize(img_src, dsize=(640, 480), interpolation=cv2.INTER_AREA)

img_src1 = cv2.imread('img/PCB1/img16.bmp', cv2.IMREAD_COLOR)
#img_src1 = cv2.resize(img_src1, dsize=(640, 480), interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_src1, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

my_color = (0, 255, 0)
text_color = (255, 255, 51)
thickness = 2
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
threshold = 140

img_dst = cv2.bitwise_not(img_gray)
ret, img_binary = cv2.threshold(img_dst, threshold, 255, cv2.THRESH_BINARY_INV)
mask = cv2.merge((img_binary, img_binary, img_binary))

dilate = cv2.dilate(img_binary, kernel, iterations=5)
dilate = cv2.erode(dilate, kernel, iterations=6)
mask_dilate = cv2.merge((dilate, dilate, dilate))

circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=20, minRadius=30, maxRadius=40)

for i, circle in enumerate(circles[0]):
    cv2.circle(img_src1, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 51, 153), 3)

# cv2.findContours()를 이용하여 이진화 이미지에서 윤곽선(컨투어)를 검색합니다.
# cv2.findContours(이진화 이미지, 검색 방법, 근사화 방법)을 의미합니다.
contours, hierachy = cv2.findContours(dilate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i, contour in enumerate(contours):

    # 윤곽선 길이 함수(Cv2.ArcLength)는 윤곽선의 전체 길이를 계산합니다.
    # Cv2.ArcLength(윤곽선 배열, 폐곡선 여부)로 윤곽선 길이를 계산합니다.
    # 윤곽선 넓이 함수(Cv2.ContourArea)는 윤곽선의 면적을 계산합니다.
    # Cv2.ContourArea(윤곽선 배열, 폐곡선 여부)로 윤곽선 면적을 계산합니다.
    # 폐곡선 여부는 시작점과 끝점의 연결 여부를 의미합니다.
    area = cv2.contourArea(contour)

    if area > 10000:

        # cv2.drawContours()을 이용하여 검출된 윤곽선을 그립니다.
        # cv2.drawContours(이미지, [윤곽선], 윤곽선 인덱스, (B, G, R), 두께, 선형 타입)을 의미합니다.
        cv2.drawContours(img_src1, contours, i, my_color, thickness)

        mu = cv2.moments(contour)
        cx = int(mu['m10'] / (mu['m00'] + 1e-5))
        cy = int(mu['m01'] / (mu['m00'] + 1e-5))

        cv2.circle(img_src1, (cx, cy), 5, (0, 0, 255), -1)

        #i.sort_index()
        #print(i)
        cv2.putText(img_src1, f'{i}:{int(area)}', (cx-60, cy+85), cv2.FONT_HERSHEY_COMPLEX, 0.9, text_color, 1)

        #cont = cv2.hconcat([img_src, img_grays])
        #cont1 = cv2.hconcat([mask, img_src1])
        #conts = cv2.vconcat([cont, cont1])


        # cv2.imshow('img', img_src1)

img_src1 = cv2.pyrDown(img_src1)
cv2.imshow('img_src1', img_src1)

mask = cv2.pyrDown(mask)
#cv2.imshow('mask', mask)
#cv2.imshow('mask_morp1', mask_morp1)
#cv2.imshow('mask_morp2', mask_morp2)
#cv2.imshow('mask_erode', mask_erode)
mask_dilate = cv2.pyrDown(mask_dilate)
#cv2.imshow('mask_dilate', mask_dilate)

cv2.waitKey(0)
cv2.destroyAllWindows()