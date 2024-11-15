import cv2

# 이미지 열기
img = cv2.imread("bird.jpg", cv2.IMREAD_COLOR)

cv2.imshow('img', img)
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘
