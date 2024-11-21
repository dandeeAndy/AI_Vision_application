import cv2
import numpy as np

img = cv2.imread('mb_002.jpg')
img2 = img.copy()

# 이미지 리사이즈
hor = 1700 # 변환될 가로 픽셀 사이즈
ver = 1600 # 변환될 세로 픽셀 사이즈
img = cv2.resize(img, (hor, ver)) 

# 그레이스케일과 바이너리 스케일 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# 히스토그램 평활화
#imgray = cv2.equalizeHist(imgray)

# 이미지 블러 / 이미지 샤프닝
filter_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
imgray_sharp_1 = cv2.filter2D(imgray, -1, filter_sharp)

imgray_blur = cv2.blur(imgray, (21, 21))
#imgray_blur  = cv2.GaussianBlur(imgray, (21, 21), sigmaX = 0, sigmaY = 0)

# 원본 이미지 - 블러 이미지
result = imgray - imgray_blur

# 이진화
ret, img_binary = cv2.threshold(result, 150, 255, cv2.THRESH_BINARY_INV)

# 결과 출력
cv2.imshow('imgray', imgray)
cv2.imshow('imgray_blur', imgray_blur)
cv2.imshow('imgdiff', result)

cv2.waitKey()
cv2.destroyAllWindows()