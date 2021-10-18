# 원 검출(Circle Detection)
import cv2
import numpy as np

img_src = cv2.imread('img/img18.jpg', cv2.IMREAD_COLOR)
img_src = cv2.pyrDown(img_src)

img_src1 = cv2.imread('img/img18.jpg', cv2.IMREAD_COLOR)
img_src1 = cv2.pyrDown(img_src1)

img_gray = cv2.cvtColor(img_src1, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# cv2.HoughCircles(검출 이미지, 검출 방법, 해상도 비율, 최소 거리, 캐니 엣지 임곗값, 중심 임곗값, 최소 반지름, 최대 반지름)를 이용하여 원 검출을 진행합니다.
# 검출 방법은 항상 2단계 허프 변환 방법(21HT, 그레이디언트)만 사용합니다.
# 해상도 비율은 원의 중심을 검출하는 데 사용되는 누산 평면의 해상도를 의미합니다.
# 인수를 1로 지정할 경우 입력한 이미지와 동일한 해상도를 가집니다. 즉, 입력 이미지 너비와 높이가 동일한 누산 평면이 생성됩니다.
# 또한 인수를 2로 지정하면 누산 평면의 해상도가 절반으로 줄어 입력 이미지의 크기와 반비례합니다.
# 최소 거리는 일차적으로 검출된 원과 원 사이의 최소 거리입니다. 이 값은 원이 여러 개 검출되는 것을 줄이는 역할을 합니다.
# 캐니 엣지 임곗값은 허프 변환에서 자체적으로 캐니 엣지를 적용하게 되는데, 이때 사용되는 상위 임곗값을 의미합니다.
# 하위 임곗값은 자동으로 할당되며, 상위 임곗값의 절반에 해당하는 값을 사용합니다.
# 중심 임곗값은 그레이디언트 방법에 적용된 중심 히스토그램(누산 평면)에 대한 임곗값입니다. 이 값이 낮을 경우 더 많은 원이 검출됩니다.
# 최소 반지름과 최대 반지름은 검출될 원의 반지름 범위입니다. 0을 입력할 경우 검출할 수 있는 반지름에 제한 조건을 두지 않습니다.
# 최소 반지름과 최대 반지름에 각각 0을 입력할 경우 반지름을 고려하지 않고 검출하며, 최대 반지름에 음수를 입력할 경우 검출된 원의 중심만 반환합니다.
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 50, param1=250, param2=10, minRadius=40, maxRadius=60)

for i in circles[0]:
    cv2.circle(img_src1, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), 5)

conts = cv2.hconcat([img_src, img_src1])
cv2.imshow('img', conts)

cv2.waitKey(0)
cv2.destroyAllWindows()