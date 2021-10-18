# OpenCV laplacian
# 라플라시안 함수(cv2.Laplacian)로 입력 이미지에서 가장자리를 검출할 수 있습니다.
# 라플라시안은 2차 미분의 형태로 가장자리가 밝은 부분에서 발생한 것인지, 어두운 부분에서 발생한 것인지 알 수 있습니다.
# 2차 미분 방식은 X 축과 Y 축을 따라 2차 미분한 합을 의미합니다.
# dst = cv2.laplacian(src, ddepth, ksize, scale, delta, borderType)은 입력 이미지(src)에 출력 이미지 정밀도(ddepth)를 설정하고 커널 크기(ksize), 비율(scale), 오프셋(delta), 테두리 외삽법(borderType)을 설정하여 결과 이미지(dst)를 반환합니다.
# 출력 이미지 정밀도는 반환되는 결과 이미지의 정밀도를 설정합니다.
# 커널 크기는 라플라시안 필터의 크기를 설정합니다. 커널의 값이 1일 경우, 중심값이 -4인 3 x 3 Aperture Size를 사용합니다.
# 비율과 오프셋은 출력 이미지를 반환하기 전에 적용되며, 주로 시각적으로 확인하기 위해 사용합니다.
# 픽셀 외삽법은 이미지 가장자리 부분의 처리 방식을 설정합니다.
import cv2

img_src = cv2.imread('img/image.jpg', cv2.IMREAD_COLOR)
img_src = cv2.pyrDown(img_src)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)

img_laplacian = cv2.Laplacian(img_gray, cv2.CV_16S, ksize=3)
img_laplacian = cv2.convertScaleAbs(img_laplacian)

img_laplacians = cv2.merge((img_laplacian, img_laplacian, img_laplacian))

cont = cv2.hconcat([img_src, img_grays, img_laplacians])

cv2.imshow('img', cont)
cv2.waitKey(0)
cv2.destroyAllWindows()