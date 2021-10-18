# Opencv 차선인식
import cv2
import pickle
import numpy as np

nwindows = 9
margin = 150
minpix = 1

capture = cv2.VideoCapture('mov/challenge.mp4',)

def undistort(img, cal_dir='output/output4/wide_dist_pickle.p'):
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

    # 왜곡 보정
    img_undist = undistort(img_frames)

    img_gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    #height, width = img_gray.shape[:2]

    # BGR -> HSL 변환
    img_hls = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HLS)

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
    src = np.float32([(0.23, 0.75), (0.78, 0.75), (0.1, 1), (1, 1)])
    dst = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    src = src * np.float32((width, height))
    dst = dst * np.float32(dst_size)

    M = cv2.getPerspectiveTransform(src, dst)
    img_warp = cv2.warpPerspective(img_binary_added, M, dst_size)
    img_warps = cv2.merge((img_warp, img_warp, img_warp))

    img_warp1 = img_warp.copy()

    histogram = np.sum(img_warp1[height // 2:, :], axis=0)

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

    img_warp1s = cv2.merge((img_warp1, img_warp1, img_warp1))

    cont = cv2.hconcat([img_frame, img_undist, img_grays, img_hls])
    cont1 = cv2.hconcat([img_hls_hs, img_hls_ls, img_hls_ss, img_sobel_scaleds])
    cont2 = cv2.hconcat([img_sobel_xs, img_sobel_x_abss])
    cont3 = cv2.hconcat([sx_binarys, s_binarys, img_binary_addeds, img_warp1s])
    cont4 = cv2.vconcat([cont, cont1, cont3])

    imgs = cv2.pyrDown(cont4)
    imgss = cv2.pyrDown(imgs)
    cv2.imshow('imgs', imgss)

    imgs1 = cv2.pyrDown(cont2)
    #imgss1 = cv2.pyrDown(imgs1)
    cv2.imshow('imgss1', imgs1)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 키
        break

capture.release()
cv2.destroyAllWindows()