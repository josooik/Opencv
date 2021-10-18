# OpenCV sobel
# 소벨 함수(cv2.Sobel)로 입력 이미지에서 가장자리를 검출할 수 있습니다.
# 미분 값을 구할 때 가장 많이 사용되는 연산자이며, 인접한 픽셀들의 차이로 기울기(Gradient)의 크기를 구합니다.
# 이때 인접한 픽셀들의 기울기를 계산하기 위해 컨벌루션 연산을 수행합니다.
# dst = cv2.Sobel(src, ddepth, dx, dy, ksize, scale, delta, borderType)은 입력 이미지(src)에 출력 이미지 정밀도(ddepth)를 설정하고 dx(X 방향 미분 차수), dy(Y 방향 미분 차수), 커널 크기(ksize), 비율(scale), 오프셋(delta), 테두리 외삽법(borderType)을 설정하여 결과 이미지(dst)를 반환합니다.
# 출력 이미지 정밀도는 반환되는 결과 이미지의 정밀도를 설정합니다.
# X 방향 미분 차수는 이미지에서 X 방향으로 미분할 차수를 설정합니다.
# Y 방향 미분 차수는 이미지에서 Y 방향으로 미분할 차수를 설정합니다.
# 커널 크기는 소벨 마스크의 크기를 설정합니다. 1, 3, 5, 7 등의 홀수 값을 사용하며, 최대 31까지 설정할 수 있습니다.
# 비율과 오프셋은 출력 이미지를 반환하기 전에 적용되며, 주로 시각적으로 확인하기 위해 사용합니다.
# 픽셀 외삽법은 이미지 가장자리 부분의 처리 방식을 설정합니다.
# Tip : X 방향 미분 차수와 Y 방향 미분 차수는 합이 1 이상이여야 하며, 0의 값은 해당 방향으로 미분하지 않음을 의미합니다.
import cv2

img_src = cv2.imread('img/image.jpg', cv2.IMREAD_COLOR)
img_src = cv2.pyrDown(img_src)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)

img_sobel_col = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_col = cv2.convertScaleAbs(img_sobel_col)

img_sobel_row = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel_row = cv2.convertScaleAbs(img_sobel_row)

img_sobel = cv2.addWeighted(img_sobel_col, 1.0, img_sobel_row, 1.0, 0)

img_sobel_cols = cv2.merge((img_sobel_col, img_sobel_col, img_sobel_col))
img_sobel_rows = cv2.merge((img_sobel_row, img_sobel_row, img_sobel_row))
img_sobels = cv2.merge((img_sobel, img_sobel, img_sobel))

cont = cv2.hconcat([img_src, img_grays, img_hsv])
cont1 = cv2.hconcat([img_sobel_cols, img_sobel_rows, img_sobels])
cont2 = cv2.vconcat([cont, cont1])

cv2.imshow('dst', cont2)
cv2.waitKey(0)
cv2.destroyAllWindows()