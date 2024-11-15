# 참고링크 : https://076923.github.io/posts/Python-opencv-12/
import cv2

# 이미지 열기
img = cv2.imread("mb_001.jpg", cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#=============================================
# 이미지 이진화
threhold = 255 # 이 값보다 픽셀값이 큰 영역은 흰색(255), 작으면 검정색(0)으로 변환
ret, img_binary = cv2.threshold(img_gray, threhold, 255, cv2.THRESH_BINARY)
# 임계값(threshold)을 0에서 255까지 다양하게 변화시키면서 실행해보세요.
# 이진화를 적용하면 이미지는 무조건 흰색(255값) 또는 검정색(0값)만 포함합니다.
# 변수창에서 img_binary를 더블 클릭하여 픽셀값을 직접 확인해보세요.
#=============================================

# 이미지 표시
cv2.imshow('before', img_gray) # 이진화 적용 전
cv2.imshow('after', img_binary) # 이진화 적용 후
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘