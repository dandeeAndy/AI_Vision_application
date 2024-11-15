import cv2

# 이미지 열기
img = cv2.imread("opencv_logo.png", cv2.IMREAD_COLOR) # 컬러로 열기
#img = cv2.imread("opencv_logo.png", cv2.IMREAD_GRAYSCALE) # 흑백으로 열기

# 이미지 타입과 배열 크기 확인
print("image type: ", type(img))
print("image shape: ", img.shape)

# 이미지 표시
cv2.imshow('input image', img)
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘

# 이미지 저장
cv2.imwrite('opencv_logo(save).png', img)














