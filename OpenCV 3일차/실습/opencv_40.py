# OpenCV opencv_38 + opencv_39(sobel + laplacian)
import cv2

img_src = cv2.imread('img/img13.jpg', cv2.IMREAD_COLOR)
img_src = cv2.resize(img_src, dsize=(500, 500), interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

img_sobel_col = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_col = cv2.convertScaleAbs(img_sobel_col)

img_sobel_row = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel_row = cv2.convertScaleAbs(img_sobel_row)

img_sobel = cv2.addWeighted(img_sobel_col, 1.0, img_sobel_row, 1.0, 0)

img_sobel_cols = cv2.merge((img_sobel_col, img_sobel_col, img_sobel_col))
img_sobel_rows = cv2.merge((img_sobel_row, img_sobel_row, img_sobel_row))
img_sobels = cv2.merge((img_sobel, img_sobel, img_sobel))

img_laplacian = cv2.Laplacian(img_gray, cv2.CV_16S, ksize=3)
img_laplacian = cv2.convertScaleAbs(img_laplacian)

img_laplacians = cv2.merge((img_laplacian, img_laplacian, img_laplacian))

cont = cv2.hconcat([img_src, img_grays, img_laplacians])
cont1 = cv2.hconcat([img_sobel_cols, img_sobel_rows, img_sobels])
cont2 = cv2.vconcat([cont, cont1])

cv2.imshow('dst', cont2)
cv2.waitKey(0)
cv2.destroyAllWindows()