# OpenCV Camera Calibration(카메라 왜곡 펴기)
import cv2
import glob

images = glob.glob('img/*.jpg')
total_images = len(images)

idx = 0

while True:
    fname = images[idx]
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out_str = f'{idx}/{total_images}'
    cv2.putText(img, out_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow('img', img)

    key = cv2.waitKey(0)

    if key == 27: # ESC
        break

    elif key == 0x61:
        idx -= 1

    elif key == 0x64:
        idx += 1

    if idx < 0:
        idx = 0

cv2.destroyAllWindows()