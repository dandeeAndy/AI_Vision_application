import sys
import cv2


src = cv2.imread('opencv_logo.png') # src.shape=(320, 480)


dst = cv2.flip(src, 1)
# 1은 좌우 반전, 0은 상하 반전입니다.
# -1은 상하좌우 반전

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()