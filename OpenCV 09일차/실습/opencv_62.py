# 저장된 피클 데이터를 읽어와서 왜곡 보정하기(undistort)
import cv2
import pickle

with open('output/output3/wide_dist_pickle.p', mode='rb') as f:
    file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']

img = cv2.imread('img/camera_cal/test_cal.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)

img_result = cv2.hconcat([img, dst])
img_result = cv2.pyrDown(img_result)
cv2.imshow('dsdt', img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()