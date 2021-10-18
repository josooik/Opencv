# OpenCV 차선인식 엣지 동영상 출력
import cv2
import numpy as np

capture = cv2.VideoCapture('mov/challenge.mp4',)

trap_bottom_width = 0.8
trap_top_width = 0.1
trap_height = 0.4

rho = 2
theta = 1 * np.pi / 180
threshold = 15
min_line_lenght = 10
max_line_gap = 20

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

while True:
    ret, img_frame = capture.read()

    img_frames = img_frame.copy()
    img_frames1 = img_frame.copy()
    img_frames2 = img_frame.copy()

    capture_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
    capture_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    capture_grays = cv2.cvtColor(capture_gray, cv2.COLOR_GRAY2BGR)

    # 블러(흐림)을 사용해서 노이즈 제거
    img_gauss = cv2.GaussianBlur(capture_gray, (5, 5), 0)
    img_gausss = cv2.merge((img_gauss, img_gauss, img_gauss))

    # 임계값(Threshold)를 사용하여 이진화
    _, frame_binary = cv2.threshold(img_gauss, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    frame_binarys = cv2.merge((frame_binary, frame_binary, frame_binary))

    # 외각선(엣지) 구하기 : Canny 엣지를 사용
    frame_canny = cv2.Canny(frame_binary, 50, 150)
    frame_cannys = cv2.merge((frame_canny, frame_canny, frame_canny))

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
        cv2.line(img_frames1, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 1)

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

    img_zero = np.zeros_like(frame_canny)
    height, width = img_zero.shape[:2]

    pts = np.array([[
        ((width * (1-trap_bottom_width)) // 2, height),
        ((width * (1-trap_top_width)) // 2, (1-trap_height) * height),
        (width - (width * (1-trap_top_width)) // 2, (1-trap_height) * height),
        (width -(width * (1-trap_bottom_width)) // 2, height)]],
        dtype=np.int32)

    cv2.fillPoly(img_zero, pts, 255, cv2.LINE_AA)

    img_canny_poly = cv2.bitwise_and(frame_canny, img_zero)
    img_canny_polys = cv2.merge((img_canny_poly, img_canny_poly, img_canny_poly))
    img_frame_canny_poly = cv2.bitwise_and(img_frames1, img_canny_polys)

    lines = cv2.HoughLinesP(img_canny_poly, rho, theta, threshold, minLineLength=min_line_lenght, maxLineGap=max_line_gap)

    for i, line in enumerate(lines):
        cv2.line(img_frames2, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 1)

    cont = cv2.hconcat([img_frame, capture_hsv, capture_grays, img_gausss, frame_binarys])
    cont1 = cv2.hconcat([frame_cannys, img_frames1, img_frames_poly, img_dst_y, img_dst_w])
    cont2 = cv2.hconcat([img_dst_yw, img_poly, img_canny_polys, img_frame_canny_poly, img_frames2])
    cont3 = cv2.vconcat([cont, cont1, cont2])

    img_frame = cv2.pyrDown(cont3)
    img_frame = cv2.pyrDown(img_frame)
    cv2.imshow('Video', img_frame)

    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    key = cv2.waitKey(10)
    if key == 27:  # ESC 키
        break

capture.release()
cv2.destroyAllWindows()