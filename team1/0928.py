import cv2
import numpy as np
import os

# 이미지 업로드
def load_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없음: {file_path}")
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"이미지 로드 실패: {file_path}")
    return image
#================================================================================================
# 이미지 전처리 함수
#================================================================================================
# 그레이스케일 변환
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러 적용
def gaussian_blur(image, kernel_size=(9, 9), sigma=2):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# 이진화 적용
def binary(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
           threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)

kernel_size_row = 3
kernel_size_col = 3
# 열림 연산 적용
def opening(image, kernel_size=(kernel_size_row, kernel_size_col)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 닫힘 연산 적용
def closing(image, kernel_size=(kernel_size_row, kernel_size_col)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 팽창 연산 적용
def dilation(image, kernel_size=(kernel_size_row, kernel_size_col), iterations=2):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

# 침식 연산 적용
def erosion(image, kernel_size=(kernel_size_row, kernel_size_col), iterations=2):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

# 에지 검출
def detect_edges(image, ksize=3):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    img_edge = cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0)
    return cv2.convertScaleAbs(img_edge)

# 컨투어 찾기
def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 허프 변환을 이용한 원 검출
def detect_ROI(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, 
                               param1=50, param2=30, minRadius=100, maxRadius=0)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        return largest_circle
    return None

# 원형 마스크 생성
def create_circular_mask(image, center, radius):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

def apply_ROI(image):
    original = image.copy()
    container = detect_ROI(image)
    
    if container is not None:
        # Create a circular ROI
        center = (container[0], container[1])
        radius = container[2]
        mask = create_circular_mask(image, center, radius)
        
        # Apply the mask to the image
        roi = image.copy()
        roi[~mask] = 0
        
        # Draw the ROI circle on the original image
        cv2.circle(original, center, radius, (255, 0, 0), 5)

def process_image(image_path):
    # Load the image
    image = load_image(image_path)
    original = image.copy()
    
    # Convert the image to grayscale
    gray = grayscale(image)
    
    # Apply Gaussian blur
    blurred = gaussian_blur(gray)
    
    # Find contours
    contours = find_contours(blurred)
    
    # Apply ROI
    apply_ROI(original)
    
    return original, len(contours)

def display_results(image_path):
    # Process the image
    output, count = process_image(image_path)
    
    # Display the results
    cv2.imshow('Result', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = 'mb_001.jpg'
    display_results(image_path)