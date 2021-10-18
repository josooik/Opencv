# OpenCV 웹캠으로 파란색 추출
import cv2

cap = cv2.VideoCapture(0)

while (1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 앞서 설명한 파랑색 계열의 범위
    lower_blue = (110, 100, 50)  # 자료형은 튜플형태로(H, S, V)
    upper_blue = (130, 255, 255)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    maks = cv2.merge((mask, mask, mask))

    cont = cv2.hconcat([frame, maks])
    cont1 = cv2.hconcat([cont, res])

    cv2.imshow('frame', cont1)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()