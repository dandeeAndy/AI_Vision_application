import cv2
import numpy as np

# 이미지 열기
img = cv2.imread("opencv_logo.png", cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지 이진화
threhold = 150 # 이 값보다 픽셀값이 크면 255, 작으면 0으로 변환
ret, img_binary = cv2.threshold(img_gray, threhold, 255, cv2.THRESH_BINARY)

# 크롭
img_crop = img_binary[0:230 , 200:460] # 이미지 크롭
pix_sum = np.sum(img_crop) # 픽셀값을 모두 더하기
pix_sum_cnt = (int)(pix_sum/255) # 이진화 이미지에서 흰색 픽셀의 개수에 해당

# 이미지 표시
cv2.imshow('image(orginal)', img_gray) # 원본 이미지
cv2.imshow('image(binary)', img_binary) # 이진화 이미지
cv2.imshow('image(crop)', img_crop) # 이진화 후 크롭된 이미지
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘