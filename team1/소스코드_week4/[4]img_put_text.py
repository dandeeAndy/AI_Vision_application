#import numpy as np
import cv2

# 이미지 열기
img = cv2.imread("bird.jpg", cv2.IMREAD_COLOR)

#======================================================
x = 100 # x 좌표
y = 200 # y 좌표
val = 10000 # 표시할 내용
test_size = 2 # 글자 크기
color = (0, 255, 0) # 글씨 색깔 지정하기 위해(B/G/R 순)
line = 5 # 선 굵기
cv2.putText(img,  str(val), (x, y), cv2.FONT_HERSHEY_SIMPLEX, test_size, color, line, cv2.LINE_AA)


x = 280 # x 좌표
y = 300 # y 좌표
val = "Bird" # 표시할 내용
test_size = 3 # 글자 크기
color = (255, 0, 255) # 글씨 색깔 지정하기 위해(B/G/R 순)
line = 5 # 선 굵기
cv2.putText(img,  str(val), (x, y), cv2.FONT_HERSHEY_SIMPLEX, test_size, color, line, cv2.LINE_AA)
#======================================================

# 이미지 표시
cv2.imshow('text', img)
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘