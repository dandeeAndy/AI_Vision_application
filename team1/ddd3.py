import numpy as np
import cv2

# 이미지 읽기
img = cv2.imread('mb_005.jpg', cv2.IMREAD_COLOR)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV 범위의 초기값 설정
lower_h = 0
lower_s = 0
lower_v = 0
upper_h = 180
upper_s = 255
upper_v = 255

# 트랙바의 콜백 함수 (실제로는 필요 없음)
def nothing(x):
    pass

# 트랙바를 만들고 윈도우 생성
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Lower H', 'Image', lower_h, 180, nothing)
cv2.createTrackbar('Lower S', 'Image', lower_s, 255, nothing)
cv2.createTrackbar('Lower V', 'Image', lower_v, 255, nothing)
cv2.createTrackbar('Upper H', 'Image', upper_h, 180, nothing)
cv2.createTrackbar('Upper S', 'Image', upper_s, 255, nothing)
cv2.createTrackbar('Upper V', 'Image', upper_v, 255, nothing)

while True:
    # 트랙바에서 현재 값을 읽어옴
    lower_h = cv2.getTrackbarPos('Lower H', 'Image')
    lower_s = cv2.getTrackbarPos('Lower S', 'Image')
    lower_v = cv2.getTrackbarPos('Lower V', 'Image')
    upper_h = cv2.getTrackbarPos('Upper H', 'Image')
    upper_s = cv2.getTrackbarPos('Upper S', 'Image')
    upper_v = cv2.getTrackbarPos('Upper V', 'Image')

    # HSV 범위 설정
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # 색상 범위 마스크 생성
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

    # 비트 연산을 통해 마스크 적용
    res = cv2.bitwise_and(img, img, mask=mask)

    # 이미지를 고정 비율로 리사이즈
    aspect_ratio = img.shape[1] / img.shape[0]
    new_width = 800  # 원하는 너비
    new_height = int(new_width / aspect_ratio)  # 비율에 맞게 높이 계산
    res_resized = cv2.resize(res, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 결과 출력
    cv2.imshow('Image', res_resized)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 눌러 종료
        break

cv2.destroyAllWindows()
