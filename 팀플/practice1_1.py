import cv2
import numpy as np

# scale_factor = 0.14

# images = []
# for filename in mbimg:
#     color_image = cv2.imread(filename, cv2.IMREAD_COLOR)
#     if color_image is None:
#         print(f"Error loading image {filename}. Check the file path.")
#         continue
    
#     width = int(color_image.shape[1] * scale_factor)
#     height = int(color_image.shape[0] * scale_factor)
#     resized_image = cv2.resize(color_image, (width, height))
    
#     images.append(resized_image)

# min_height = min(image.shape[0] for image in images)
# min_width = min(image.shape[1] for image in images)

# for i in range(len(images)):
#     images[i] = cv2.resize(images[i], (min_width, min_height))

# rows = 2
# cols = 5
# combined_image = np.zeros((min_height * rows, min_width * cols, 3), dtype=np.uint8)

# for i in range(len(images)):
#     row = i // cols
#     col = i % cols
#     combined_image[row * min_height:(row + 1) * min_height, col * min_width:(col + 1) * min_width] = images[i]

# cv2.imshow('Combined Images', combined_image)
# cv2.waitKey(0)  # 키 입력 대기
# cv2.destroyAllWindows()  # 창 닫기

#=========================================================================================================================

#=========================================================================================================================



def load_image(file_path):
    return cv2.imread(file_path)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(9, 9), sigma=2):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_adaptive_threshold(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)

def apply_opening(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def detect_edges(image, ksize=3):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_combined = cv2.sqrt(sobel_x**2 + sobel_y**2)
    return cv2.convertScaleAbs(sobel_combined)

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def visualize_cotton_swabs(image, contours):
    output = image.copy()
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(output, center, radius, (0, 255, 0), 2)
    
    cotton_swab_count = len(contours)
    cv2.putText(output, f'Cotton Swab Count: {cotton_swab_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return output, cotton_swab_count

def process_image(image_path):
    # 이미지 로드
    image = load_image(image_path)
    
    # 그레이스케일 변환
    gray = convert_to_grayscale(image)
    
    # 가우시안 블러 적용
    blurred = apply_gaussian_blur(gray)
    
    # 적응형 이진화 적용
    adaptive_thresh = apply_adaptive_threshold(blurred)
    
    # 열림 연산 적용
    opened = apply_opening(adaptive_thresh)
    
    # 소벨 연산자를 이용한 에지 검출
    edges = detect_edges(opened)
    
    # 컨투어 검출
    contours = find_contours(edges)
    
    # 면봉 머리 시각화 및 개수 계산
    output, cotton_swab_count = visualize_cotton_swabs(image, contours)
    
    return {
        'original': image,
        'gray': gray,
        'blurred': blurred,
        'adaptive_thresh': adaptive_thresh,
        'opened': opened,
        'edges': edges,
        'output': output,
        'cotton_swab_count': cotton_swab_count
    }

def display_results(results):
    # 결과 창 크기 설정 및 이미지 출력
    window_names = [
        'Original Image', 'Grayscale', 'Blurred', 'Adaptive Thresholding',
        'Opened Image', 'Edge Detection', 'Detected Cotton Swabs'
    ]
    
    for name in window_names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 800, 800)
    
    cv2.imshow('Original Image', results['original'])
    cv2.imshow('Grayscale', results['gray'])
    cv2.imshow('Blurred', results['blurred'])
    cv2.imshow('Adaptive Thresholding', results['adaptive_thresh'])
    cv2.imshow('Opened Image', results['opened'])
    cv2.imshow('Edge Detection', results['edges'])
    cv2.imshow('Detected Cotton Swabs', results['output'])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 메인 실행 부분
if __name__ == "__main__":
    image_paths = ["mb_001.jpg", "mb_002.jpg", "mb_003.jpg", "mb_004.jpg", "mb_005.jpg", 
          "mb_006.jpg", "mb_007.jpg", "mb_008.jpg", "mb_010.jpg", "mb_011.jpg", "mb_plus.jpg"]  # 처리할 이미지 경로 리스트
    
    for image_path in image_paths:
        results = process_image(image_path)
        print(f"Image: {image_path}, Cotton Swab Count: {results['cotton_swab_count']}")
        display_results(results)