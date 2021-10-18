# OpenCV 유튜브 동영상에서 상하 좌우 반전
# 이미지 역상/ 반전 255->0 / 0->255로
import cv2
import numpy as np

capture = cv2.VideoCapture('mov/mov1.mp4',)

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

while True:
    ret, img_frame = capture.read()

    img_frame = cv2.resize(img_frame, dsize=(654, 480), interpolation=cv2.INTER_AREA)
    img_frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    img_frame_grays = cv2.cvtColor(img_frame_gray, cv2.COLOR_GRAY2BGR)

    img_dst = cv2.bitwise_not(img_frame)

    img_flip_lr = cv2.flip(img_frame, 1)  # 1 : 좌우대칭 / 0 : 상하대칭 / -1 : 상하좌우 대칭
    img_flip_ud = cv2.flip(img_frame, 0)
    img_flip_lrud = cv2.flip(img_frame, -1)

    cont = cv2.hconcat([img_frame, img_flip_lr])
    cont1 = cv2.hconcat([cont, img_frame_grays])

    cont2 = cv2.hconcat([img_flip_ud, img_flip_lrud])
    cont3 = cv2.hconcat([cont2, img_dst])

    cont4 = cv2.vconcat([cont1, cont3])

    cv2.imshow('Video', cont4)


    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    key = cv2.waitKey(20)
    if key == 27:  # ESC 키
        break

capture.release()
cv2.destroyAllWindows()