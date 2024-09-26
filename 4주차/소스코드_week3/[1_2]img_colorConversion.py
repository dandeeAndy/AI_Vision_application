import cv2

# 이미지 열기
img = cv2.imread("zebra.jpg", cv2.IMREAD_COLOR) # 컬러로 열기
cv2.imshow("1.image(bgr)", img) # BGR 이미지 표시

img_copied = img.copy() # 이미지 카피

# BGR 이미지를 Grayscale 이미지로 변환
img_gray = cv2.cvtColor(img_copied, cv2.COLOR_BGR2GRAY)
cv2.imshow("2.image(gray)", img_gray) # Grayscale 이미지 표시

# BGR 이미지를 RGB 이미지로 변환
img_rgb = cv2.cvtColor(img_copied, cv2.COLOR_BGR2RGB)
cv2.imshow("3.image(rgb)", img_rgb) # RGB 이미지 표시

# RGB 이미지를 R / G / B 이미지로 쪼개기
(img_r, img_g, img_b) = cv2.split(img_rgb)
cv2.imshow("4.image(r)", img_r) # R 이미지 표시

# R / G / B 이미지를 합쳐서 다시 RGB로 만들기
img_merged = cv2.merge([img_b, img_g, img_r])
cv2.imshow("5.image(merged)", img_merged) # RGB 이미지 표시

cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘
