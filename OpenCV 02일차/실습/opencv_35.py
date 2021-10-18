# opencv_26 + opencv_34 영상 합침
import cv2

capture = cv2.VideoCapture('mov/mov1.mp4',)

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

while True:
    ret, img_frame = capture.read()

    img_frame = cv2.resize(img_frame, dsize=(554, 250), interpolation=cv2.INTER_AREA)
    capture_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)

    capture_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    capture_grays = cv2.cvtColor(capture_gray, cv2.COLOR_GRAY2BGR)

    img_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
    img_h, img_S, img_v = cv2.split(img_hsv)

    # HSV로 초록색 정보를 좀 더 구체적으로 표시
    lower_yellow = (20, 125, 130)  # 자료형은 튜플형태로(H, S, V)
    upper_yellow = (30, 255, 255)  # 자료형은 튜플형태로(H, S, V)

    lower_blue = (100, 100, 30)  # 자료형은 튜플형태로(H, S, V)
    upper_blue = (130, 255, 255)  # 자료형은 튜플형태로(H, S, V)

    lower_green = (50, 100, 100)  # 자료형은 튜플형태로(H, S, V)
    upper_green = (70, 255, 255)  # 자료형은 튜플형태로(H, S, V)

    img_mask_y = cv2.inRange(img_hsv, lower_yellow, upper_yellow)  # 노랑색 정보 추출(특정 범위 안에 있는 행렬 원소 검출)
    img_mask_b = cv2.inRange(img_hsv, lower_blue, upper_blue)  # 파랑색 정보 추출(특정 범위 안에 있는 행렬 원소 검출)
    img_mask_g = cv2.inRange(img_hsv, lower_green, upper_green)  # 파랑색 정보 추출(특정 범위 안에 있는 행렬 원소 검출)

    img_dst_y = cv2.bitwise_and(img_frame, img_frame, mask=img_mask_y)  # AND 비트연산
    img_dst_b= cv2.bitwise_and(img_frame, img_frame, mask=img_mask_b)  # AND 비트연산
    img_dst_g = cv2.bitwise_and(img_frame, img_frame, mask=img_mask_g)  # AND 비트연산

    img_mask_ys = cv2.merge((img_mask_y, img_mask_y, img_mask_y))
    img_mask_bs = cv2.merge((img_mask_b, img_mask_b, img_mask_b))
    img_mask_gs = cv2.merge((img_mask_g, img_mask_g, img_mask_g))

    ret1, mask = cv2.threshold(capture_grays, 100, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_dst = cv2.bitwise_not(img_frame)

    img_flip_lr = cv2.flip(img_frame, 1)  # 1 : 좌우대칭 / 0 : 상하대칭 / -1 : 상하좌우 대칭
    img_flip_ud = cv2.flip(img_frame, 0)
    img_flip_lrud = cv2.flip(img_frame, -1)

    cont = cv2.hconcat([img_frame, capture_grays])
    cont1 = cv2.hconcat([cont, mask_inv])

    cont2 = cv2.hconcat([img_dst_y, img_dst_b])
    cont3 = cv2.hconcat([cont2, img_dst_g])

    cont4 = cv2.hconcat([img_frame, img_flip_lr])
    cont5 = cv2.hconcat([cont4, capture_grays])

    cont6 = cv2.hconcat([img_flip_ud, img_flip_lrud])
    cont7 = cv2.hconcat([cont6, img_dst])

    cont8 = cv2.vconcat([cont1, cont3])
    cont9 = cv2.vconcat([cont5, cont7])

    cont10 = cv2.vconcat([cont9, cont8])

    cv2.imshow('Video', cont10)

    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    key = cv2.waitKey(20)
    if key == 27:  # ESC 키
        break

capture.release()
cv2.destroyAllWindows()