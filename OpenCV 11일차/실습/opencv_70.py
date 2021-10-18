# Opencv 차선인식
import cv2
import pickle
import numpy as np

nwindows = 9
margin = 150
minpix = 1

trap_bottom_width = 0.8
trap_top_width = 0.1
trap_height = 0.4

road_width = 2.5  # 도로 폭

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

capture = cv2.VideoCapture('mov/challenge.mp4',)

play_mode = 1 # 0: play once 1:play continuously

if capture.isOpened() == False:
  print("카메라를 열 수 없습니다.")
  exit(1)

video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
codec = cv2.VideoWriter_fourcc('m','p','4','v') # .mp4
#codec = cv2.VideoWriter_fourcc('M','J','P','G') # .avi

fps = 30.0
# 동영상 파일을 저장하려면 VideoWrite객체를 생성
# VideoWriter객체를 초기화 하기 위해 저장할 동영상 파일 이름,
# 코덱, 프레임레이트, 이미지 크기를 지정해야함
writer = cv2.VideoWriter('output/mov/output1.mp4', codec, fps, (width,height))
writer1 = cv2.VideoWriter('output/mov/process1.mp4', codec, fps, (5120, 2880), isColor=True)

#VideoWriter객체를 성공적으로 초기화 했는지 체크
if writer.isOpened() == False:
   print('동영상 저장파일객체 생성하는데 실패하였습니다.')
   exit(1)

if writer1.isOpened() == False:
   print('동영상 저장파일객체 생성하는데 실패하였습니다.')
   exit(1)

#Esc키를 눌러 동영상을 중단하면 종료직전까지 동영상이 저장됨
video_counter = 0

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

def undistort(img, cal_dir='output/output3/wide_dist_pickle.p'):
    # cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

while True:
    ret, img_frame = capture.read()

    if img_frame is None:
        break

    img_frames = img_frame.copy()
    img_frames1 = img_frame.copy()
    img_frames2 = img_frame.copy()

    # 왜곡 보정
    #img_undist = undistort(img_frames)

    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    #img_gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # BGR -> HSL 변환
    img_hls = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HLS)
    #img_hls = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HLS)

    img_hls_h, img_hls_l, img_hls_s = cv2.split(img_hls)
    img_hls_hs = cv2.merge((img_hls_h, img_hls_h, img_hls_h))
    img_hls_ls = cv2.merge((img_hls_l, img_hls_l, img_hls_l))
    img_hls_ss = cv2.merge((img_hls_s, img_hls_s, img_hls_s))

    #  소벨 필터 적용
    img_sobel_x = cv2.Sobel(img_hls_l, cv2.CV_64F, 1, 1)
    img_sobel_xs = cv2.merge((img_sobel_x, img_sobel_x, img_sobel_x))

    img_sobel_x_abs = abs(img_sobel_x)
    img_sobel_x_abss = cv2.merge((img_sobel_x_abs, img_sobel_x_abs, img_sobel_x_abs))

    img_sobel_scaled = np.uint8(img_sobel_x_abs * 255 / np.max(img_sobel_x_abs))
    img_sobel_scaleds = cv2.merge((img_sobel_scaled, img_sobel_scaled, img_sobel_scaled))

    sx_threshold = (15, 255)
    sx_binary = np.zeros_like(img_sobel_scaled)
    sx_binary[(img_sobel_scaled >= sx_threshold[0]) & (img_sobel_scaled <= sx_threshold[1])] = 255
    sx_binarys = cv2.merge((sx_binary, sx_binary, sx_binary))

    s_threshold = (100, 255)
    s_binary = np.zeros_like(img_hls_s)
    s_binary[(img_hls_s >= s_threshold[0]) & (img_hls_s <= s_threshold[1])] = 255
    s_binarys = cv2.merge((s_binary, s_binary, s_binary))

    img_binary_added = cv2.addWeighted(sx_binary, 1.0, s_binary, 1.0, 0)
    img_binary_addeds = cv2.merge((img_binary_added, img_binary_added, img_binary_added))

    height, width = img_binary_added.shape[:2]

    dst_size = (width, height)
    src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    src = src * np.float32((width, height))

    #cv2.line(src, (0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1), (0, 0, 255), 2)

    dst = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    dst = dst * np.float32(dst_size)

    M = cv2.getPerspectiveTransform(src, dst)
    img_warp = cv2.warpPerspective(img_binary_added, M, dst_size)
    img_warps = cv2.merge((img_warp, img_warp, img_warp))

    img_warp1 = cv2.warpPerspective(img_frames1, M, dst_size)

    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)

    # axis=0->x축 즉 x축의 모든 값을 sum(더한다)는 의미
    histogram = np.sum(img_warp[height // 2:, :], axis=0)

    # axis=0->x축 즉 x축의 모든 값을 sum(더한다)는 의미
    midpoint = len(histogram) // 2

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(height / nwindows)

    # 영상에서 0이 앙닌 모든 점의 x,y점을 정의
    nonzero = img_warp.nonzero()
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

        cv2.rectangle(img_warp1, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (100, 255, 255), 3)
        cv2.rectangle(img_warp1, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (100, 255, 255), 3)

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

        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]

    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    # 곡선이므로 2차방정식 기준에 의해 계수들 구함
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    # 10개 데이터에 대해 평균값을 이용함으로써 중간의 튀는 값을 막음
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # x와 y값을 그리기 위해 생성
    # 0부터 height-1(99)까지 height(100)개 만큼 1차원 배열 만들기
    ploty = np.linspace(0, height - 1, height)

    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

    # 선에 색칠하기
    img_warp1[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 100]
    img_warp1[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 100, 255]

    color_img = np.zeros_like(img_warp1)
    left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((left, right))

    mid = (left_fitx + right_fitx) // 2

    left_fix_mean = np.mean(left_fitx)
    right_fix_mean = np.mean(right_fitx)

    mid_points = np.array([np.transpose(np.vstack((mid, ploty)))])

    mean_mid = np.mean(mid)

    # 차선 그리기
    cv2.polylines(color_img, np.int_(points), False, (0, 255, 255), 10)

    # 차선 안쪽 채우기
    cv2.fillPoly(color_img, np.int_(points), (0, 255, 0))

    # 원본영상과 차선검출 영상 합치기 : 가중치 조절(1:100%, 0.4:40%)
    img_warp2 = cv2.addWeighted(img_warp1, 0.4, color_img, 0.4, 0)

    cv2.line(img_warp2, (width // 2, 0), (width // 2, height), (0, 0, 255), 20)
    cv2.polylines(img_warp2, np.int_(mid_points), False, (255, 255, 128), 20)
    cv2.circle(img_warp2, (np.int_(mean_mid), height // 2), 10, (255, 0, 255), 15)
    cv2.line(img_warp2, (width // 2, height // 2), (np.int_(mean_mid), height // 2), (255, 255, 255), 20)
    cv2.imshow('img_warp2', img_warp2)

    # inverse버전
    M1 = cv2.getPerspectiveTransform(dst, src)  # src: 4개의 원본 좌표점  dst: 4개의 결과 좌표점 '

    # src의 좌표점 4개를 dst의 좌표점으로 인식한다. 저 경우에는 각 점을 줄이겠다는 의미(그만큼 축소)
    img_warp3 = cv2.warpPerspective(img_warp2, M1, dst_size)

    img_f = cv2.addWeighted(img_frames1, 1, img_warp3, 0.4, 0)

    road_width_pixel = road_width / (right_fix_mean - left_fix_mean) # pixel 1 = m
    error = width // 2 - mean_mid # 도로중심과 이미지 중심이 떨어진 픽셀 거리
    dis_error = error * road_width_pixel # 도로중심과 이미지 중심이 떨어진 거리

    if(dis_error < 0):
        cv2.putText(img_f, 'right : %.2fm' %(abs(dis_error)), (550, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    elif (dis_error > 0):
        cv2.putText(img_f, 'left : %.2fm' %(abs(dis_error)), (550, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    else:
        cv2.putText(img_f, 'center : %.2fm' %(abs(dis_error)), (550, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cont = cv2.hconcat([img_frame, img_frame, img_grays, img_hls])
    cont1 = cv2.hconcat([img_hls_hs, img_hls_ls, img_hls_ss, img_sobel_scaleds])
    cont2 = cv2.hconcat([img_sobel_xs, img_sobel_x_abss])
    cont3 = cv2.hconcat([sx_binarys, s_binarys, img_binary_addeds, img_warps])
    cont4 = cv2.hconcat([img_warp1, img_warp2, img_warp3, img_f])
    cont5 = cv2.vconcat([cont, cont1, cont3, cont4])

    writer.write(img_f)
    writer1.write(cont5)

    imgs = cv2.pyrDown(cont5)
    imgs = cv2.pyrDown(imgs)
    cv2.imshow('imgs', imgs)
    cv2.imshow('img_f', img_f)

    imgs1 = cv2.pyrDown(cont2)
    imgs1 = cv2.pyrDown(imgs1)
    #cv2.imshow('imgss1', imgs1)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 키
        break

    if video_counter == video_length:
        video_counter = 0
    else:
        video_counter += 1

    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

capture.release()
writer.release()
writer1.release()
cv2.destroyAllWindows()