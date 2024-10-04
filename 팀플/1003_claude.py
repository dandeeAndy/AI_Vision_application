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
    
    step_y = (height - roi_height) / 3
    step_x = (width - roi_width) / 3
    
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
    # kernel_size = (5, 5)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # dilation = cv2.dilate(gray, np.ones(kernel_size, np.uint8), iterations=2)
    # erosion = cv2.erode(dilation, np.ones(kernel_size, np.uint8), iterations=2)
    edges = cv2.GaussianBlur(gray, (5, 5), 0)
    # print("shape: ", edges.shape)
    # edges = cv2.Canny(blurred, 50, 150)
    return edges

# 면봉 검출 함수
def detect_cotton_swabs(roi, edges, min_radius, max_radius, param1, param2):
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.3, minDist=50,
                               param1=param1, param2=param2, 
                               minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return []

# 이미지 처리 함수
def process_image(image_path, min_radius, max_radius, param1, param2):
    image = load_image(image_path)
    if image is None:
        print("이미지를 불러오는 데 실패했습니다.")
        return
    
    height, width = image.shape[:2]
    roi_size = (int(height * 5/17), int(width * 5/17))
    roi_list = create_roi(image, roi_size)
    
    # 면봉 검출
    total_cotton_swabs = 0
    preprocessed_images = []
    
    for idx, (roi, (x, y)) in enumerate(roi_list):
        edges = preprocess_roi(roi)
        preprocessed_images.append(edges)
        
        circles = detect_cotton_swabs(roi, edges, min_radius, max_radius, param1, param2)
        total_cotton_swabs += len(circles)
        
        for (cx, cy, r) in circles:
            cv2.circle(image, (x + cx, y + cy), r, (0, 0, 255), 2)
        
        cv2.rectangle(image, (x, y), (x + roi_size[1], y + roi_size[0]), (0, 255, 0), 2)
        cv2.putText(image, f"Count: {len(circles)}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    print(f"검출된 총 면봉 개수: {total_cotton_swabs}")
    cv2.imwrite("result.jpg", image)
    
    # 결과 이미지와 전처리된 이미지 표시
    plt.figure(figsize=(20, 10))
    plt.subplot(4, 5, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Result Image")
    plt.axis('off')
    
    for i, preprocessed_img in enumerate(preprocessed_images):
        plt.subplot(4, 5, i+2)
        plt.imshow(preprocessed_img, cmap='gray')
        plt.title(f"ROI {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    root = tk.Tk()
    root.withdraw()
    
    image_path = filedialog.askopenfilename()
    # min_radius = int(input("최소 반지름 입력 (기본값 10): ") or 10)
    # max_radius = int(input("최대 반지름 입력 (기본값 30): ") or 30)
    # param1 = int(input("param1 입력 (기본값 50): ") or 50)
    # param2 = int(input("param2 입력 (기본값 30): ") or 30)
    min_radius = 30
    max_radius = 50
    param1 = 40
    param2 = 28
    
    process_image(image_path, min_radius, max_radius, param1, param2)

if __name__ == "__main__":
    main()