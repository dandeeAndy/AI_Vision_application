import cv2
import numpy as np
import os

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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")
    return image

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
    cv2.putText(output, f'Count: {cotton_swab_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return output, cotton_swab_count

def visualize_cotton_swabs_improved(image, contours, min_radius=5, max_radius=1000, roi=None):
    output = image.copy()
    valid_count = 0
    
    # height, width = image.shape[:2]
    # print(height, width)
    if roi is None:
        roi = [100, 100, 1900, 1900]  # [x, y, w, h]
    
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 원의 크기 제한 적용
        if min_radius <= radius <= max_radius:
            # ROI 내에 있는지 확인
            if (roi[0] <= center[0] <= roi[0] + roi[2] and 
                roi[1] <= center[1] <= roi[1] + roi[3]):
                cv2.circle(output, center, radius, (0, 255, 0), 2)
                valid_count += 1
    
    cv2.putText(output, f'Count: {valid_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.rectangle(output, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)
    
    return output, valid_count


def process_image(image_path):
    try:
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
        output, cotton_swab_count = visualize_cotton_swabs_improved(image, contours)
        
        return output, cotton_swab_count
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, 0
    
def display_results(images, cotton_swab_counts):
    rows = 2
    cols = 5
    scale_factor = 0.16
    
    resized_images = []
    for img in images:
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        resized_images.append(cv2.resize(img, (width, height)))
    
    min_height = min(image.shape[0] for image in resized_images)
    min_width = min(image.shape[1] for image in resized_images)
    
    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (min_width, min_height))
    
    combined_image = np.zeros((min_height * rows, min_width * cols, 3), dtype=np.uint8)
    
    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols
        combined_image[row * min_height:(row + 1) * min_height, col * min_width:(col + 1) * min_width] = img
    
    for i, count in enumerate(cotton_swab_counts):
        row = i // cols
        col = i % cols
        text_pos = (col * min_width + 10, (row + 1) * min_height - 10)
        cv2.putText(combined_image, f"Count: {count}", text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Cotton Swab Detection Results', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_paths = ["mb_001.jpg", "mb_002.jpg", "mb_003.jpg", "mb_004.jpg", "mb_005.jpg",
                   "mb_006.jpg", "mb_007.jpg", "mb_008.jpg", "mb_010.jpg", "mb_011.jpg"]
    
    processed_images = []
    cotton_swab_counts = []
    
    for image_path in image_paths:
        output, count = process_image(image_path)
        if output is not None:
            processed_images.append(output)
            cotton_swab_counts.append(count)
            print(f"Image: {image_path}, Cotton Swab Count: {count}")
    
    display_results(processed_images, cotton_swab_counts)
