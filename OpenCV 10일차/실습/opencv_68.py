import cv2
import numpy as np

nwindows = 9
margin = 150
minpix = 1

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

img_src = cv2.imread('img/img20.PNG', cv2.IMREAD_COLOR)
img_binary = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
height, width = img_binary.shape[:2]

left_fit_ = np.empty(3)
right_fit_ = np.empty(3)

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

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx  = nonzero_x[left_lane_inds]
lefty = nonzero_y[left_lane_inds]

rightx = nonzero_x[right_lane_inds]
righty = nonzero_y[right_lane_inds]

left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

left_a.append(left_fit[0])
left_b.append(left_fit[1])
left_c.append(left_fit[2])

right_a.append(right_fit[0])
right_b.append(right_fit[1])
right_c.append(right_fit[2])

left_fit_[0] = np.mean(left_a[-10:])
left_fit_[1] = np.mean(left_b[-10:])
left_fit_[2] = np.mean(left_c[-10:])

right_fit_[0] = np.mean(right_a[-10:])
right_fit_[1] = np.mean(right_b[-10:])
right_fit_[2] = np.mean(right_c[-10:])

# x와 y값을 그리기 위해 생성
# 0부터 height-1(99)까지 height(100)개 만큼 1차원 배열 만들기
ploty = np.linspace(0, height-1, height)
left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]
img_src[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 100]
img_src[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 100, 255]


color_img = np.zeros_like(img_src)
left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
points = np.hstack((left, right))

# 차선 그리기
cv2.polylines(color_img, np.int_(points), False, (0, 255, 255),10)

# 차선 안쪽 채우기
cv2.fillPoly(color_img, np.int_(points), (0, 255, 0))

# 원본영상과 차선검출 영상 합치기 : 가중치 조절(1:100%, 0.4:40%)
img_src = cv2.addWeighted(img_src, 1, color_img, 0.4, 0)

cv2.imshow('dst', img_src)
cv2.waitKey(0)
cv2.destroyAllWindows()