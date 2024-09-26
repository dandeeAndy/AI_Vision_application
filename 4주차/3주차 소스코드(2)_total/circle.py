import cv2
import numpy as np


# 원이 있는 이미지를 올림.

img1 = cv2.imread('circle.jpg') # minRadius=50, maxRadius=100)
img2 = img1.copy()

# 0~9까지 가우시안 필터로 흐리게 만들어 조절함.
img2 = cv2.GaussianBlur(img2, (9, 9), 0)
# 그레이 이미지로 바꿔서 실행해야함.
imgray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow('before', imgray)


# 원본과 비율 / 찾은 원들간의 최소 중심거리 / param1, param2를 조절해 원을 찾음
# circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=12, minRadius=200, maxRadius=350)
circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 100, param1 = 250, param2 = 10, minRadius = 80, maxRadius = 500)

if circles is not None:
    circles = np.uint16(np.around(circles))

    print(circles)

    for i in circles[0, :]:
        cv2.circle(img2, (i[0], i[1]), round(i[2]*0.2), (0, 0, 0), 2)


    cv2.imshow('HoughCircle', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('원을 찾을 수 없음')

