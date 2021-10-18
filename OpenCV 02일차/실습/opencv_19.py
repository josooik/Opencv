# RGB to HSV 코드
import cv2 as cv
import numpy as np

B = 255
G = 0
R = 0

my_color = np.uint8([[[B, G, R]]]) # BGR
my_hsv = cv.cvtColor(my_color, cv.COLOR_BGR2HSV)

print(f"H : {my_hsv[0][0][0]} S : {my_hsv[0][0][1]} V : {my_hsv[0][0][2]}")