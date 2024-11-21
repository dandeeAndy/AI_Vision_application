import cv2
import numpy as np
import os

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"파일이 존재하지 않습니다: {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러오는 데 실패했습니다: {image_path}")
        return None
    print(f"이미지를 성공적으로 불러왔습니다. 크기: {image.shape}")
    return image

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

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = resize_image(blurred, width=800)
    print(f"전처리된 이미지 크기: {blurred.shape}")
    return blurred

# 가장 큰 원 검출 함수(거의 확정)
def detect_largest_circle(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=1000,
                               param1=40, param2=100, minRadius=800, maxRadius=1200)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = circles[0][0]
        return largest_circle
    return None

def create_mask(image, center, radius):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center[0], center[1]), radius - 5, 255, -1)
    return mask

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def main(image_path):
    # 이미지 불러오기
    image = load_image(image_path)
    
    # 이미지 전처리
    preprocessed = preprocess_image(image)
    
    # 가장 큰 원 검출
    largest_circle = detect_largest_circle(preprocessed)
    
    if largest_circle is not None:
        center = (largest_circle[0], largest_circle[1])
        radius = largest_circle[2]
        
        # 마스크 생성
        mask = create_mask(image, center, radius)
        
        # 마스크 적용
        result = apply_mask(image, mask)
        
        # 결과 표시
        cv2.imshow('Original Image', image)
        cv2.imshow('Masked Image', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("원을 찾을 수 없습니다.")

# 이미지 경로를 지정하여 실행
# image_path = 'mb_002.jpg'  # 실제 이미지 경로로 변경해주세요

image_paths = ["mb_001.jpg", "mb_002.jpg", "mb_003.jpg", "mb_004.jpg", "mb_005.jpg",
                   "mb_006.jpg", "mb_007.jpg", "mb_008.jpg", "mb_010.jpg", "mb_011.jpg"]


for image_path in image_paths:
    main(image_path)