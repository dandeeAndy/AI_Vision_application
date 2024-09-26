import cv2
import numpy as np

# 이미지 열기
#img = cv2.imread("opencv_logo.png", cv2.IMREAD_COLOR)
img = cv2.imread("com.jpg", cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gray = 255 - img_gray

#==========================================================
# 모폴로지 연산(침식 or 팽창)
kernel_size_row = 3 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (설정값 : 3 or 5 or 7)
kernel_size_col = 3 # 이 값이 커질수록 침식 또는 팽창 효과가 강해집니다 (설정값 : 3 or 5 or 7)
# 3, 5, 7등으로 값을 바꿔서 실행해보세요. cv2.erode()나 cv.dilate를 반복하여 적용하는 것도 가능합니다.
kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

erosion_image = cv2.erode(img_gray, kernel, iterations=4)  # 침식연산이 적용된 이미지
                                          # iterations를 2 이상으로 설정하면 반복 적용됨
dilation_image = cv2.dilate(img_gray, kernel, iterations=8)  # 팽창연산이 적용된 이미지
                                                             # iterations를 2 이상으로 설정하면 반복 적용됨

open_image = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel) # 열림연산 : 침식 후 팽창
close_image = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel) # 닫힘연산 : 팽창 후 침식
#==========================================================

# 이미지 표시
cv2.imshow('1.img_gray', img_gray) # 원본
cv2.imshow('2.erosion_image', erosion_image) # 침식
cv2.imshow('3.dilation_image', dilation_image) # 팽창
cv2.imshow('4.open_image', open_image) # 열림(침식 후 팽창)
cv2.imshow('5.close_image', close_image) # 닫힘(팽창 후 침식)
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘