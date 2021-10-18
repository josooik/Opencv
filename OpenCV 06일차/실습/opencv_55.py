# OpenCV 차선인식 엣지 이미지 출력
import cv2
import numpy as np

img_frame = cv2.imread('img/img19.png', cv2.IMREAD_COLOR)
img_frames = img_frame.copy()
img_gray = cv2.cvtColor(img_frames, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# 블러(흐림)을 사용해서 노이즈 제거
img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_gausss = cv2.merge((img_gauss, img_gauss, img_gauss))

# 임계값(Threshold)를 사용하여 이진화
_, frame_binary = cv2.threshold(img_gauss, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
frame_binarys = cv2.merge((frame_binary, frame_binary, frame_binary))

# 외각선(엣지) 구하기 : Canny 엣지를 사용
frame_canny = cv2.Canny(frame_binary, 50, 150)
frame_cannys = cv2.merge((frame_canny, frame_canny, frame_canny))

rho = 2
theta = 1 * np.pi / 180
threshold = 15
min_line_lenght = 10
max_line_gap = 20

# 허프 변환
# cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap) → lines
# image – 8bit, single-channel binary image, canny edge를 선 적용.
# rho – r 값의 범위 (0 ~ 1 실수)
# theta – 𝜃 값의 범위(0 ~ 180 정수)
# threshold – 만나는 점의 기준, 숫자가 작으면 많은 선이 검출되지만 정확도가 떨어지고, 숫자가 크면 정확도가 올라감.
# minLineLength – 선의 최소 길이. 이 값보다 작으면 reject.
# maxLineGap – 선과 선사이의 최대 허용간격. 이 값보다 작으며 reject.
lines = cv2.HoughLinesP(frame_canny, rho, theta, threshold, minLineLength=min_line_lenght, maxLineGap=max_line_gap)

for i, line in enumerate(lines):
    cv2.line(img_frames, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 1)

cont = cv2.hconcat([img_frame, img_grays, img_gausss])
cont1 = cv2.hconcat([frame_binarys, frame_cannys, img_frames])
cont2 = cv2.vconcat([cont, cont1])

img_frame1 = cv2.pyrDown(cont2)
cv2.imshow("img_frame", img_frame1)
cv2.waitKey(0)
cv2.destroyAllWindows()