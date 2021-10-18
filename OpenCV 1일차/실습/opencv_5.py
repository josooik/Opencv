# OpenCV 유튜브 동영상 출력
import cv2

capture = cv2.VideoCapture('mov/mov.mp4',)

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

while True:
    ret, img_frame = capture.read()

    '''
    if ret == False:  # 동영상이 끝까지 재생
        print("동영상 읽기 완료")
        break
    '''
    cv2.imshow('Video', img_frame)

    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    key = cv2.waitKey(20)
    if key == 27:  # ESC 키
        break

capture.release()
cv2.destroyAllWindows()