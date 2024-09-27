import cv2
import numpy as np

# 이미지 읽기 및 원본 비율 유지하면서 크기 조정 함수
def resize_image(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# 이미지 읽기 및 크기 조정
image_path = 'mb_004.jpg'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img_resized = resize_image(img, width=800)  # 화면에 맞게 크기 조정

# 그레이스케일 변환
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# 블러 적용
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 팽창과 침식 연산으로 이미지 전처리
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(blurred, kernel, iterations=2)
eroded = cv2.erode(dilated, kernel, iterations=1)

# 허프 변환을 이용한 원 검출
circles = cv2.HoughCircles(
    eroded, 
    cv2.HOUGH_GRADIENT, 
    dp=1.3, 
    minDist=16,  # 면봉들이 밀접해 있기 때문에 최소 거리 조정
    param1=40, 
    param2=25,  # 더 많은 원을 검출하도록 임계값 조정
    minRadius=4,  # 면봉의 크기에 맞는 최소 반지름 조정
    maxRadius=16  # 면봉의 크기에 맞는 최대 반지름 조정
)

# 검출된 원을 이미지에 그리기 및 개수 세기
num_circles = 0
if circles is not None:
    circles = np.uint16(np.around(circles))
    num_circles = len(circles[0])
    for circle in circles[0, :]:
        center = (circle[0], circle[1])  # 원의 중심
        radius = circle[2] - 3  # 원의 크기 줄이기 (반지름에서 3을 뺌)
        # 원 그리기 (빨간색, 두께 2로 조정)
        cv2.circle(img_resized, center, radius, (0, 0, 255), 2)  # 두께 2로 조정

# 이미지 위에 원의 개수 출력
cv2.putText(img_resized, f"Count: {num_circles}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

# 결과 출력
cv2.imshow('Detected Circles', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
