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
def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
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
def erosion(image, kernel_size=(kernel_size_row, kernel_size_col), iterations=1):
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

import cv2
import numpy as np

def process_image(file_path):
    # 이미지 로드
    image = load_image(file_path)
    
    # 그레이스케일 변환
    gray_image = grayscale(image)
    
    # 가우시안 블러 적용
    blurred_image = gaussian_blur(gray_image)
    
    # # 이진화 적용
    # binary_image = binary(blurred_image)
    
    # # 열림 연산 적용
    # opened_image = opening(binary_image)
    
    # # 닫힘 연산 적용
    # closed_image = closing(opened_image)
    
    # 팽창 연산 적용
    dilated_image = dilation(blurred_image)
    
    # 침식 연산 적용
    eroded_image = erosion(dilated_image)
    
    # # 에지 검출
    # edged_image = detect_edges(dilated_image)
    
    # # 컨투어 찾기
    # contours = find_contours(edged_image)
    
    process_image = eroded_image
    
    # # 가장 큰 원 찾기
    # largest_circle = find_largest_circle(eroded_image, image)
    # if largest_circle is None:
    #     return image, None
    
    # # 면봉 검출 및 이미지에 표시
    # processed_image = mark_cotton_buds(image, largest_circle)
    
    return process_image

def find_largest_circle(image, original_image):
    # 에지 검출
    edged_image = detect_edges(image)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    
    if largest_contour is None:
        return None
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    return (int(x), int(y)), int(radius)

# def mark_cotton_buds(image, largest_circle):
#     # 이미지에 원 표시
#     x, y, r = largest_circle
#     cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    
#     # 면봉 개수 출력
#     cv2.putText(image, f"면봉 수: 1", (x, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
#     return image

# def mask_image(image, largest_circle):
#     mask = np.zeros_like(image)
#     cv2.circle(mask, largest_circle[0], largest_circle[1], (255, 255, 255), -1)
#     return cv2.bitwise_and(image, mask)

# def get_roi(image, largest_circle):
#     x, y = largest_circle[0]
#     radius = largest_circle[1]
#     roi = image[y-radius:y+radius, x-radius:x+radius]
#     if roi.shape[0] == 0 or roi.shape[1] == 0:
#         return None
#     return roi

def detect_circles_in_roi(roi_image):
    circles = cv2.HoughCircles(roi_image, cv2.HOUGH_GRADIENT, 1.3, 16,
                              param1=40, param2=25, minRadius=4, maxRadius=16)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

def display_results(image, circles):
    # 이미지에 원 표시
    for circle in circles[0]:
        x, y, r = circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    
    # 면봉 개수 출력
    cv2.putText(image, f"면봉 수: {len(circles[0])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 결과 이미지 출력
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 이미지 파일 경로
    file_path = "mb_001.jpg"
    
    # 이미지 처리 및 결과 출력
    circles = process_image(file_path)
    
    display_results(masked_image, circles)