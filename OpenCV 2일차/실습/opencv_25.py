# RED의 경우 Hue 영역이 떨어져서 존재하므로 두영역을 찾고 합침
import cv2
img_src = cv2.imread('img/img10.jpg', cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(1500, 300), interpolation=cv2.INTER_AREA)

# 이미지를 BGR에서 HSV로 색변환
img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV) # HSV

# HSV에서 RED는 2개의 영역이 존재
s_min = 50; s_max = 255
v_min = 50; v_max = 255

lower_red1 = (0, s_min, v_min)   # 자료형은 튜플형태로(H,S,V)
upper_red1 = (7, s_max, v_max)   # 자료형은 튜플형태로(H,S,V)

lower_red2 = (165, s_min, v_min) # 자료형은 튜플형태로(H,S,V)
upper_red2 = (180, s_max, v_max) # 자료형은 튜플형태로(H,S,V)

img_mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1) # 빨강정보1(0~7) 추출
img_mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2) # 빨강정보2(165~180) 추출
img_mask = cv2.addWeighted(img_mask1, 1.0, img_mask2, 1.0, 0.0)
# cv2.addWeighted() => 가중치 합, 평균 연산
# src1: (입력) 첫 번째 영상
# alpha: 첫 번째 영상 가중치
# src2: 두 번째 영상. src1과 같은 크기 & 같은 타입
# beta: 두 번째 영상 가중치
# gamma: 결과 영상에 추가적으로 더할 값
# dst: 가중치 합 결과 영상
# dtype: 출력 영상(dst)의 타입

img_dst = cv2.bitwise_and(img_src, img_src, mask=img_mask)

img_maskS = cv2.merge((img_mask, img_mask, img_mask))

cont = cv2.vconcat([img_src, img_maskS])
cont1 = cv2.vconcat([cont, img_dst])

cv2.imshow("img_red", cont1)

cv2.waitKey(0)
cv2.destroyAllWindows()
