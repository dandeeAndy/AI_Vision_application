# 라이브러리 불러오기
import cv2

# 이미지 열기
filename = "mb_001.jpg" # 입력 이미지 파일명을 적으세요.
img = cv2.imread(filename, cv2.IMREAD_COLOR)


#========알고리즘 및 시각화 소스코드 작성 (시작)=========
 
# [알고리즘 구현(예시)]
mb_count = 482 # 면봉 개수(알고리즘에 의한 예측값)


# [시각화 구현(예시)]

# 시각화 결과
img_vis = cv2.putText(img, str(mb_count), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
img_vis = cv2.rectangle(img, (640, 260), (720, 340), (0, 255, 0), 3) # 면봉 검출을 사각형으로 시각화 하는 경우
img_vis = cv2.circle(img, (800, 300), 40, (0, 266, 0), 3) # 면봉 검출을 원으로 시각화 하는 경우

#========알고리즘 및 시각화 소스코드 작성 (끝)=========




# 시각화 결과 표시(예측 결과 확인용, 이 부분은 수정하지 마시오)
cv2.imshow('visualization', img_vis) # 시각화
cv2.waitKey(0) 
cv2.destroyAllWindows()
