# OpenCV 초록색 색상 이미지 추출
import cv2

img_src = cv2.imread("img/img10.jpg", cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(1500, 400), interpolation=cv2.INTER_AREA)

# 이미지를 BGR에서 HSV로 색변환
img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
img_h, img_S, img_v = cv2.split(img_hsv)

# 원하는 color의 h정보를 적용 : 초록색을 추출 ( 90도~150도(대략) )
# Opencv에서는 0~180까지만 사용 초록정보를 나누기 2함 (45~75 (대략))
img_h = cv2.inRange(img_h, 46, 75)  # 초록색 정보 추출(특정 범위 안에 있는 행렬 원소 검출)
hsv_green = cv2.bitwise_and(img_hsv, img_hsv, mask=img_h)  #  AND 비트연산
img_dst = cv2.cvtColor(hsv_green, cv2.COLOR_HSV2BGR)

cont = cv2.vconcat([img_src, img_dst])

cv2.imshow("img_green", cont)
cv2.waitKey(0)
cv2.destroyAllWindows()