import cv2
import numpy as np

# 이미지 로드
image = cv2.imread('mb_001.jpg')

# 1. 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 가우시안 블러 적용 (노이즈 제거)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# 3. 적응형 이진화 적용 (영역별 문턱값 처리)
adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 4. 열림 연산 적용 (침식 후 팽창)
kernel = np.ones((5, 5), np.uint8)  # 5x5 커널을 사용
opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

# 5. 소벨 연산자를 이용한 에지 검출 (수평과 수직 방향)
sobel_x = cv2.Sobel(opened, cv2.CV_64F, 1, 0, ksize=3)  # 수평 방향 에지
sobel_y = cv2.Sobel(opened, cv2.CV_64F, 0, 1, ksize=3)  # 수직 방향 에지

# 절대값을 취하여 두 방향의 에지 결합
sobel_combined = cv2.sqrt(sobel_x**2 + sobel_y**2)
sobel_combined = cv2.convertScaleAbs(sobel_combined)

# 6. 컨투어 검출
contours, _ = cv2.findContours(sobel_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 면봉 머리 개수 출력
cotton_swab_count = len(contours)
print(f"면봉 머리 개수: {cotton_swab_count}")

# 면봉 머리 시각화
output = image.copy()
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(output, center, radius, (0, 255, 0), 2)

# 면봉 개수 텍스트 추가
cv2.putText(output, f'Cotton Swab Count: {cotton_swab_count}', (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# 결과 창 크기 설정
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL) 
cv2.namedWindow('gray', cv2.WINDOW_NORMAL) 
cv2.namedWindow('blurred', cv2.WINDOW_NORMAL)  
cv2.namedWindow('Adaptive Thresholding Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Opened Image (after Open Operation)', cv2.WINDOW_NORMAL)
cv2.namedWindow('Sobel Edge Detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('Detected Cotton Swabs', cv2.WINDOW_NORMAL)

# 창 크기 조절
cv2.resizeWindow('Original Image', 800, 800)
cv2.resizeWindow('gray', 800, 800)
cv2.resizeWindow('blurred', 800, 800)
cv2.resizeWindow('Adaptive Thresholding Image', 800, 800)
cv2.resizeWindow('Opened Image (after Open Operation)', 800, 800)
cv2.resizeWindow('Sobel Edge Detection', 800, 800)
cv2.resizeWindow('Detected Cotton Swabs', 800, 800)

# 결과 이미지 출력
cv2.imshow('Original Image', image)   # 원본 이미지 출력
cv2.imshow('gray', gray)   # 그레이 스케일
cv2.imshow('blurred', blurred)  #가우시안 블러
cv2.imshow('Adaptive Thresholding Image', adaptive_thresh)  # 적응형 문턱값 처리 후 출력
cv2.imshow('Opened Image (after Open Operation)', opened)  # 열림 연산 적용 후 출력
cv2.imshow('Sobel Edge Detection', sobel_combined)  # 소벨 연산자 에지 검출 결과 출력
cv2.imshow('Detected Cotton Swabs', output)  # 면봉 머리 검출 결과 출력

cv2.waitKey(0)
cv2.destroyAllWindows()
