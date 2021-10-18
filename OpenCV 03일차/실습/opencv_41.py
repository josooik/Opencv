# 캐니 함수(cv2.Canny)로 입력 이미지에서 가장자리를 검출할 수 있습니다.
# 캐니 엣지는 라플라스 필터 방식을 개선한 방식으로 x와 y에 대해 1차 미분을 계산한 다음, 네 방향으로 미분합니다.
# 네 방향으로 미분한 결과로 극댓값을 갖는 지점들이 가장자리가 됩니다.
# 앞서 설명한 가장자리 검출기보다 성능이 월등히 좋으며 노이즈에 민감하지 않아 강한 가장자리를 검출하는 데 목적을 둔 알고리즘입니다.
# dst = cv2.Canny(src, threshold1, threshold2, apertureSize, L2gradient)는 입력 이미지(src)를 하위 임곗값(threshold1), 상위 임곗값(threshold2), 소벨 연산자 마스크 크기(apertureSize), L2 그레이디언트(L2gradient)을 설정하여 결과 이미지(dst)를 반환합니다.
# 하위 임곗값과 상위 임곗값으로 픽셀이 갖는 최솟값과 최댓값을 설정해 검출을 진행합니다.
# 픽셀이 상위 임곗값보다 큰 기울기를 가지면 픽셀을 가장자리로 간주하고, 하위 임곗값보다 낮은 경우 가장자리로 고려하지 않습니다.
# 소벨 연산자 마스크 크기는 소벨 연산을 활용하므로, 소벨 마스크의 크기를 설정합니다.
# L2 그레이디언트는 L2-norm으로 방향성 그레이디언트를 정확하게 계산할지, 정확성은 떨어지지만 속도가 더 빠른 L1-norm으로 계산할지를 선택합니다.
from __future__ import print_function
import cv2
import argparse

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

def CannyThreshold(val):
   low_threshold = val
   img_blur = cv2.blur(src_gray, (3, 3))
   detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
   mask = detected_edges != 0
   dst = src * (mask[:,:,None].astype(src.dtype))
   cv2.imshow(window_name, dst)

parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='img/image.jpg')
args = parser.parse_args()
src = cv2.imread(cv2.samples.findFile(args.input))

if src is None:
   print('Could not open or find the image: ', args.input)
   exit(0)

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)

CannyThreshold(0)
cv2.waitKey(0)
cv2.destroyAllWindows()