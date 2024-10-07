import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# 이미지 불러오기 함수
def load_image(image_path):
    return cv2.imread(image_path)

# ROI 생성 함수
def create_roi(image, roi_size, overlap=0.2):
    height, width = image.shape[:2]
    roi_height, roi_width = roi_size
    
    step_y = roi_height * (1 - overlap)
    step_x = roi_width * (1 - overlap)
    
    roi_list = []
    for i in range(4):
        for j in range(4):
            y = int(i * step_y)
            x = int(j * step_x)
            y = min(y, height - roi_height)
            x = min(x, width - roi_width)
            roi = image[y:y+roi_height, x:x+roi_width]
            roi_list.append((roi, (x, y)))
    
    return roi_list

# ROI 전처리 함수
def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# 면봉 검출 함수
def detect_cotton_swabs(roi, blurred, min_radius, max_radius, param1, param2):
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.3, minDist=50,
                               param1=param1, param2=param2, 
                               minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        return np.round(circles[0, :]).astype("int")
    return []

# 중복된 원 제거 함수
def remove_duplicate_circles(circles, distance_threshold):
    if len(circles) == 0:
        return []

    # 모든 원의 좌표를 하나의 리스트로 변환
    all_circles = [(x, y, r) for roi_circles in circles for (x, y, r) in roi_circles]
    
    # 중복 제거된 원들을 저장할 리스트
    unique_circles = []
    
    while len(all_circles) > 0:
        # 현재 원을 기준으로 설정
        current_circle = all_circles.pop(0)
        unique_circles.append(current_circle)
        
        # 현재 원과 가까운 원들을 제거
        all_circles = [circle for circle in all_circles
                       if calculate_distance(current_circle, circle) > distance_threshold]
    
    return unique_circles

# 원 사이의 거리 계산 함수
def calculate_distance(circle1, circle2):
    x1, y1, _ = circle1
    x2, y2, _ = circle2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# 이미지 처리 함수
def process_image(image_path, min_radius, max_radius, param1, param2, distance_threshold):
    image = load_image(image_path)
    if image is None:
        print("이미지를 불러오는 데 실패했습니다.")
        return
    
    height, width = image.shape[:2]
    roi_size = (int(height * 5/17), int(width * 5/17))
    roi_list = create_roi(image, roi_size)
    
    # 모든 ROI에서 검출된 원들을 저장할 리스트
    all_circles = []
    preprocessed_images = []
    
    for roi, (x, y) in roi_list:
        pre = preprocess_roi(roi)
        preprocessed_images.append(pre)
        
        circles = detect_cotton_swabs(roi, pre, min_radius, max_radius, param1, param2)
        # 원의 좌표를 전체 이미지 기준으로 변환
        circles = [(cx + x, cy + y, r) for (cx, cy, r) in circles]
        all_circles.append(circles)
        # ROI 영역 체크(초록색 사각형)
        cv2.rectangle(image, (x, y), (x + roi_size[1], y + roi_size[0]), (0, 255, 0), 2)
    # 중복 원 제거
    unique_circles = remove_duplicate_circles(all_circles, distance_threshold)
    
    # 결과 이미지에 원 그리기
    for (x, y, r) in unique_circles:
        cv2.circle(image, (x, y), r, (0, 0, 255), 2)
    
    total_cotton_swabs = len(unique_circles)
    print(f"검출된 총 면봉 개수: {total_cotton_swabs}")
    cv2.imwrite("result.jpg", image)
    
    # 결과 이미지와 전처리된 이미지 표시
    plt.figure(figsize=(20, 10))
    
    # 결과 이미지 표시
    plt.subplot(4, 5, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Result Image (Total: {total_cotton_swabs})")
    plt.axis('off')
    
    # ROI 이미지 표시 (오른쪽으로 정렬)
    for i, preprocessed_img in enumerate(preprocessed_images):
        row = i // 4
        col = i % 4
        plt.subplot(4, 5, 5 * row + col + 2)
        plt.imshow(preprocessed_img, cmap='gray')
        plt.title(f"ROI {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    cv2.imshow("Result", image)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.withdraw()
    
    image_path = filedialog.askopenfilename()
    min_radius = 30
    max_radius = 50
    param1 = 40
    param2 = 28
    distance_threshold = 30  # 이 값을 조정하여 중복 제거의 민감도를 조절할 수 있습니다
    
    process_image(image_path, min_radius, max_radius, param1, param2, distance_threshold)

if __name__ == "__main__":
    main()