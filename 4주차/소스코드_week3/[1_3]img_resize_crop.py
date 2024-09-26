import cv2

# 이미지 열기
img = cv2.imread("opencv_logo.png", cv2.IMREAD_COLOR)

# BGR 이미지를 Grayscale 이미지로 변환
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("1.image(gray)", img_gray) # Grayscale 이미지 표시

# 이미지 리사이즈(크기 변환)
hor = 800 # 변환될 가로 픽셀 사이즈
ver = 800 # 변환될 세로 픽셀 사이즈
img_gray_re = cv2.resize(img_gray, (hor, ver)) 
cv2.imshow("2.image(resize)", img_gray_re) # 크기 변환된 이미지 표시

# 이미지 크롭
img_crop = img_gray_re[620:801 , 50:751] # 
cv2.imshow('3. image(crop)', img_crop)

cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘












