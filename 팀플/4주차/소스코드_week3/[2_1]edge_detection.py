import cv2

# 이미지 열기
img = cv2.imread("zebra.jpg", cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#==========================================================
# 수직선 방향의 윤곽(에지) 검출
img_edge_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
img_edge_x = cv2.convertScaleAbs(img_edge_x)

# 수평선 방향의 윤곽(에지) 검출
img_edge_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_edge_y = cv2.convertScaleAbs(img_edge_y)

# 전 방향의 윤곽(에지) 검출(위 두 개의 결과를 합침)
img_edge = cv2.addWeighted(img_edge_x, 1, img_edge_y, 1, 0)
#==========================================================

# 이미지 표시
cv2.imshow('img_edge_x(vertical)', img_edge_x) # 수직선 에지
cv2.imshow('img_edge_y(horizontal)', img_edge_y) # 수평선 에지
cv2.imshow('img_edge(combined)', img_edge) # 두 결과를 합친 것
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘