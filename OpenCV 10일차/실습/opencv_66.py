import cv2
import numpy as np

nwindows = 9
margin = 150
minpix = 1

img_src = cv2.imread('img/img20.PNG', cv2.IMREAD_COLOR)
img_binary = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
height, width = img_binary.shape[:2]

img_zeros = np.zeros((600, 800, 3), dtype=np.uint8)
img_zeros[300:, :] = 255  # 이미지의 height의 절반부터 255로 만듬.

histogram = np.sum(img_binary[height//2:, :], axis=0)

midpoint = len(histogram) // 2
print(histogram)
print(midpoint)

leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

window_height = int(height / nwindows)

# 영상에서 0이 앙닌 모든 점의 x,y점을 정의
nonzero = img_binary.nonzero()
nonzero_y = np.array(nonzero[0])
nonzero_x = np.array(nonzero[1])

# 왼쪽, 오른쪽 차선의 nonzero index를 받기위해 리스트 생성
left_lane_inds = []
right_lane_inds = []

leftx_current = leftx_base
rightx_current = rightx_base

for window in range(nwindows):
    win_y_low = height - (window + 1) * window_height
    win_y_high = height - window * window_height

    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin

    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    cv2.rectangle(img_src, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (100, 255, 255), 3)
    cv2.rectangle(img_src, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (100, 255, 255), 3)

    # 아래 조건을 만족하는 점들의 인덱스 값을 리턴함
    good_left_inds = ((nonzero_y >= win_y_low) &
                      (nonzero_y < win_y_high) &
                      (nonzero_x >= win_xleft_low) &
                      (nonzero_x < win_xleft_high)).nonzero()[0]

    good_right_inds = ((nonzero_y >= win_y_low) &
                       (nonzero_y < win_y_high) &
                       (nonzero_x >= win_xright_low) &
                       (nonzero_x < win_xright_high)).nonzero()[0]

    # 리스트에 조건을 만족하는 인덱스 값을 append
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    # if you found > minpix 픽셀의 개수가 minpix보다 크면 사각형의 센터값 업데이트
    if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzero_x[good_left_inds]))
        print(leftx_current)

    if len(good_right_inds) > minpix:
        rightx_current = int(np.mean(nonzero_x[good_right_inds]))


cv2.imshow('dst', img_src)
cv2.waitKey(0)
cv2.destroyAllWindows()