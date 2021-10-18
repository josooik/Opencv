# OpenCV 텍스트 출력
import cv2
import numpy as np

height = 600
width = 800

img_zero = np.zeros((height, width, 3), dtype=np.uint8)

cv2.putText(img_zero, 'JoSooIk', (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
cv2.putText(img_zero, "Simplex", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
cv2.putText(img_zero, "Duplex", (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
cv2.putText(img_zero, "Simplex", (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,250))
cv2.putText(img_zero, "Complex Small", (50, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
cv2.putText(img_zero, "Complex", (50, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
cv2.putText(img_zero, "Triplex", (50, 260), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
cv2.putText(img_zero, "Complex", (200, 260), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,255))
cv2.putText(img_zero, "Script Simplex", (50, 330), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 255, 255))
cv2.putText(img_zero, "Script Complex", (50, 370), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 255))
cv2.putText(img_zero, "Plain Italic", (50, 430), cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 1, (255, 255, 255))
cv2.putText(img_zero, "Complex Italic", (50, 470), cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 1, (255, 255, 255))

cv2.imshow('dst', img_zero)
cv2.waitKey(0)
cv2.destroyAllWindows()