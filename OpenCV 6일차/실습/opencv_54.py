# Opencv 차선인식(opencv_52 + opencv_53)
import cv2
import numpy as np

capture = cv2.VideoCapture('mov/challenge.mp4',)

trap_bottom_width = 2.5
trap_top_width = 0.1
trap_height = 0.4

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

while True:
    ret, img_frame = capture.read()

    img_frames = img_frame.copy()
    capture_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)

    capture_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    capture_grays = cv2.cvtColor(capture_gray, cv2.COLOR_GRAY2BGR)

    img_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
    img_h, img_S, img_v = cv2.split(img_hsv)

    # HSV로 노랑색 정보를 좀 더 구체적으로 표시
    lower_yellow = (20, 125, 130)  # 자료형은 튜플형태로(H, S, V)
    upper_yellow = (40, 255, 255)  # 자료형은 튜플형태로(H, S, V)

    # HSV로 하얀색 정보를 좀 더 구체적으로 표시
    img_dst_w = np.copy(img_frame)

    bgr_threshold = [200, 200, 200]

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (img_frame[:, :, 0] < bgr_threshold[0]) \
                 | (img_frame[:, :, 1] < bgr_threshold[1]) \
                 | (img_frame[:, :, 2] < bgr_threshold[2])
    img_dst_w[thresholds] = [0, 0, 0]

    img_mask_y = cv2.inRange(img_hsv, lower_yellow, upper_yellow)  # 노랑색 정보 추출(특정 범위 안에 있는 행렬 원소 검출)
    img_mask_ys = cv2.merge((img_mask_y, img_mask_y, img_mask_y))
    img_dst_y = cv2.bitwise_and(img_frame, img_frame, mask=img_mask_y)  # AND 비트연산

    img_dst_yw = cv2.addWeighted(img_dst_y, 1.0, img_dst_w, 1.0, 0)

    ret1, mask = cv2.threshold(capture_grays, 200, 255, cv2.THRESH_BINARY)

    img_zero = np.zeros_like(img_frames)
    height, width = img_zero.shape[:2]

    pts = np.array([[
        ((width * (1-trap_bottom_width)) // 2, height),
        ((width * (1-trap_top_width)) // 2, (1-trap_height) * height),
        (width - (width * (1-trap_top_width)) // 2, (1-trap_height) * height),
        (width -(width * (1-trap_bottom_width)) // 2, height)]],
        dtype=np.int32)

    cv2.fillPoly(img_zero, pts, (255, 255, 255), cv2.LINE_AA)

    img_frames_poly = cv2.bitwise_and(img_frames, img_zero)
    img_poly = cv2.bitwise_and(img_dst_yw, img_zero)

    cont = cv2.hconcat([img_frame, capture_grays, capture_hsv])
    cont1 = cv2.hconcat([img_dst_y, img_dst_w, img_dst_yw])
    cont2 = cv2.hconcat([mask, img_frames_poly, img_poly])
    cont3 = cv2.vconcat([cont, cont1, cont2])

    img_frame = cv2.pyrDown(cont3)
    cv2.imshow('Video', img_frame)

    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    key = cv2.waitKey(10)
    if key == 27:  # ESC 키
        break

capture.release()
cv2.destroyAllWindows()