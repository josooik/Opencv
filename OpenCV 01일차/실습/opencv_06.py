# OpenCV 유튜브 동영상 여러색 채널 출력
import cv2
import numpy as np

capture = cv2.VideoCapture('mov/ironman.mp4')

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

while True:
    ret, img_frame = capture.read()
    img_frame = cv2.resize(img_frame, dsize=(540, 380), interpolation=cv2.INTER_AREA)
    capture_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    capture_grays = cv2.cvtColor(capture_gray, cv2.COLOR_GRAY2BGR)

    img_b, img_g, img_r = cv2.split(img_frame)  # 채널 분리
    height, width = img_frame.shape[:2]

    zero = np.zeros((height, width, 1), dtype=np.uint8)
    zeros = cv2.merge((zero, zero, zero))

    img_bs = cv2.merge((img_b, zero, zero))
    img_gs = cv2.merge((zero, img_g, zero))
    img_rs = cv2.merge((zero, zero, img_r))

    cont = cv2.hconcat([img_frame, capture_grays])
    cont1 = cv2.hconcat([cont, zeros])

    cont2 = cv2.hconcat([img_bs, img_gs])
    cont3 = cv2.hconcat([cont2, img_rs])

    cont4 = cv2.vconcat([cont1, cont3])

    '''
    if ret == False:  # 동영상이 끝까지 재생
        print("동영상 읽기 완료")
        break
    '''
    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.imshow('Video', cont4)

    key = cv2.waitKey(20)
    if key == 27:  # ESC 키
        break

capture.release()
cv2.destroyAllWindows()