import numpy as np
import cv2

# 이미지 열기
img = cv2.imread("bird.jpg", cv2.IMREAD_COLOR) 
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HSV 컬러 이미지로 변환

# 추출할 컬러에 대한 픽셀 범위 지정(최소/최대값)
lower_blue = np.array([80, 50, 50])     # H / S / V
upper_blue = np.array([120, 255, 255])   # H / S / V

# 추출된 컬러를 흰색 영역으로 표시한 이미지
mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

# bit 연산자를 통해 추출된 컬러만 표시한 이미지
res = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('1.input image', img)
cv2.imshow('2.binary image', mask)
cv2.imshow('3.color extraction', res)

cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘