# 도형의 모멘트 구해서 표시하기
import cv2

img_src = cv2.imread('img/img16.png', cv2.IMREAD_COLOR)
img_src1 = cv2.imread('img/img16.png', cv2.IMREAD_COLOR)

img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

my_color = (0, 255, 0)
text_color = (255, 0, 0)
thickness = 2

ret, img_binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
mask = cv2.merge((img_binary, img_binary, img_binary))

# 검출하려고 하는 도형의 외곽선 검출 : findContours()함수 사용
# cv2.findContours(이진화 이미지, 검색 방법, 근사화 방법)

# * 검색 방법
# cv2.RETR_EXTERNAL : 외곽 윤곽선만 검출하며, 계층 구조를 구성하지 않습니다.
# cv2.RETR_LIST : 모든 윤곽선을 검출하며, 계층 구조를 구성하지 않습니다.
# cv2.RETR_CCOMP : 모든 윤곽선을 검출하며, 계층 구조는 2단계로 구성합니다.
# cv2.RETR_TREE : 모든 윤곽선을 검출하며, 계층 구조를 모두 형성합니다. (Tree 구조)

# * 근사화 방법
# cv2.CHAIN_APPROX_NONE : 윤곽점들의 모든 점을 반환합니다.
# cv2.CHAIN_APPROX_SIMPLE : 윤곽점들 단순화 수평, 수직 및 대각선 요소를 압축하고 끝점만 남겨 둡니다.
# cv2.CHAIN_APPROX_TC89_L1 : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.
# cv2.CHAIN_APPROX_TC89_KCOS : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.

contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# cv2.drawContours()을 이용하여 검출된 윤곽선을 그립니다.
# cv2.drawContours(이미지, [윤곽선], 윤곽선 인덱스, (B, G, R), 두께, 선형 타입)을 의미합니다.
# 윤곽선은 검출된 윤곽선들이 저장된 Numpy 배열입니다.
# 윤곽선 인덱스는 검출된 윤곽선 배열에서 몇 번째 인덱스의 윤곽선을 그릴지를 의미합니다.
# Tip : 윤곽선 인덱스를 0으로 사용할 경우 0 번째 인덱스의 윤곽선을 그리게 됩니다. 하지만, 윤곽선 인수를 대괄호로 다시 묶을 경우, 0 번째 인덱스가 최댓값인 배열로 변경됩니다.
# Tip : 동일한 방식으로 [윤곽선], 0과 윤곽선, -1은 동일한 의미를 갖습니다. (-1은 윤곽선 배열 모두를 의미)

for i, contour in enumerate(contours):

   # contourArea()함수로 객체의 면적을 구하고 면적기준으로 임계값보다
   # 외곽선이 감싸는 영역의 면적을 반환합니다.
   area = cv2.contourArea(contour)

   # 큰 객체만 외곽선을 그리고 면적정보를 표시한다.
   if area > 100:
      cv2.drawContours(img_src1, contours, i, my_color, thickness)

      # 모멘트 그리기(무게중심)
      mu = cv2.moments(contour)
      cx = int(mu['m10'] / (mu['m00'] + 1e-5))
      cy = int(mu['m01'] / (mu['m00'] + 1e-5))

      cv2.circle(img_src1, (cx, cy), 5, (0, 255, 255), -1)
      cv2.putText(img_src1, f'{i}: {int(area)}', (cx-50, cy-20), cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 1)

      cont = cv2.hconcat([img_src, img_grays, mask, img_src1])

      cv2.imshow('img', cont)

      cv2.waitKey(0)

cv2.destroyAllWindows()