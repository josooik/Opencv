# 저장된 피클 데이터를 읽어와서 왜곡 보정하기(undistort)
import pickle
import numpy as np
import glob
import cv2

def undistort_img():
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros((6 * 9, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []
    # Get directory for all calibration images
    images = glob.glob('img/camera_cal/calibration*.jpg')

    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)

    # Test undistortion on img
    img_size = (img.shape[1], img.shape[0])

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('output/output4/wide_dist_pickle.p', 'wb'))


def undistort(img, cal_dir='output/output4/wide_dist_pickle.p'):
    # cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

capture = cv2.VideoCapture('mov/t5.mp4')

while True:
    ret, img_frame = capture.read()

    if img_frame is None:
        break

    img_frame1 = img_frame.copy()

    calibration = undistort(img_frame1)

    cont = cv2.hconcat([img_frame, calibration])

    #videos = cv2.pyrDown(cont)
    cv2.imshow('videos', cont)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 키
        break

capture.release()
cv2.destroyAllWindows()