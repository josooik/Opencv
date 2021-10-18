# OpenCV 더 정밀한 노랑색 이미지 추출
# inRange( ) 함수를 이용하여 HSV 색채널을 이용한 색상 검출이 가능하다.
# bitwise_and( ) 함수를 이용하여 색상 검출 된 영역을 컬러로 표현할 수 있다.
# erode 와 dilate 를 이용하여 검출된 이미지에서 노이즈를 제거할 수 있다.
import cv2

img_src = cv2.imread("img/img10.jpg", cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(1500, 300), interpolation=cv2.INTER_AREA)

# 이미지를 BGR에서 HSV로 색변환
img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
img_h, img_S, img_v = cv2.split(img_hsv)

# HSV로 초록색 정보를 좀 더 구체적으로 표시
lower_yellow = (20, 125, 130)  # 자료형은 튜플형태로(H, S, V)
upper_yellow = (60, 255, 255 )  # 자료형은 튜플형태로(H, S, V)

img_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)  # 노랑색 정보 추출(특정 범위 안에 있는 행렬 원소 검출)
img_dst = cv2.bitwise_and(img_src, img_src, mask=img_mask)   # AND 비트연산

img_maskS = cv2.merge((img_mask, img_mask, img_mask))

cont = cv2.vconcat([img_src, img_maskS])
cont1 = cv2.vconcat([cont, img_dst])

cv2.imshow("img_yellow", cont1)

cv2.waitKey(0)
cv2.destroyAllWindows()