import numpy as np
import cv2

img = cv2.imread('mb_001.jpg', cv2.IMREAD_COLOR) #BGR
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #HSV로 이미지 변경
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grayscale

blurred = cv2.GaussianBlur(hsv_img, (21, 21), 0)  # 가우시안블러

lower_blue = np.array([0, 0, 110])     # H / S / V
upper_blue = np.array([40, 120, 160])   # H / S / V

# 추출된 컬러를 흰색 영역으로 표시한 이미지
mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

# bit 연산자를 통해 추출된 컬러만 표시한 이미지
res = cv2.bitwise_and(img, img, mask=mask)



cv2.namedWindow('Image', cv2.WINDOW_NORMAL) #창 수동조절
cv2.resizeWindow('Image', 800, 800) # 창 크기 조절
cv2.imshow('Image', res) #이미지 보여주기

cv2.waitKey(0)
cv2.destroyAllWindows()