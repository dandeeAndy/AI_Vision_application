import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

# 파일 탐색기를 통해 이미지 경로를 선택하는 함수
def select_image():
    root = tk.Tk()
    root.withdraw()  # Tkinter 창 숨기기
    image_path = filedialog.askopenfilename(title="Select an Image", 
                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return image_path

# 이미지 불러오기 함수
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러오지 못했습니다. 경로를 확인해주세요: {image_path}")
    return image

# 평균 밝기 계산 함수
def calculate_brightness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]  # V 채널 (밝기)
    average_brightness = np.mean(v_channel)
    return average_brightness

# CLAHE 적용 함수 (밝기 기반)
def apply_clahe_based_on_brightness(image, brightness_threshold=100):
    average_brightness = calculate_brightness(image)
    
    if average_brightness > brightness_threshold:
        print(f"밝은 사진입니다. 밝기: {average_brightness}")
        # 밝은 사진에 대해 CLAHE 강하게 적용
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    else:
        print(f"어두운 사진입니다. 밝기: {average_brightness}")
        # 어두운 사진에 대해 CLAHE 약하게 적용
        clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return result

# 면봉통 외곽 검출 및 마스킹 함수 (컨투어 사용)
def mask_outside_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(image)  # 마스크는 이미지와 같은 크기여야 함
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 마스크를 그레이스케일로 변환
        mask = mask.astype(np.uint8)  # 마스크를 CV_8U 타입으로 변환
        result = cv2.bitwise_and(image, image, mask=mask)  # 마스크를 적용한 이미지
        return result, mask
    return image, None

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

# V 히스토그램을 분석하여 최적의 마스크 범위를 계산하는 함수
def get_v_mask_range(v_channel):
    hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
    v_peak_idx = np.argmax(hist[100:250]) + 100
    lower_v = max(v_peak_idx - 60, 0)
    upper_v = min(v_peak_idx + 40, 255)
    return lower_v, upper_v

# ROI 전처리 함수
def preprocess_roi(roi):
    hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_img)
    lower_v, upper_v = get_v_mask_range(v_channel)
    lower_bound = np.array([0, 0, lower_v])
    upper_bound = np.array([180, 255, upper_v])
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    res = cv2.bitwise_and(roi, roi, mask=mask)
    gray_blurred = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_blurred, (5, 5), 0)
    return blurred_roi

# 면봉 검출 함수 (마스킹 적용)
def detect_cotton_swabs(roi, blurred, min_radius, max_radius, param1, param2, mask):
    masked_blurred = cv2.bitwise_and(blurred, blurred, mask=mask)
    circles = cv2.HoughCircles(masked_blurred, cv2.HOUGH_GRADIENT, dp=1.3, minDist=60,
                                param1=param1, param2=param2, 
                                minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        return np.round(circles[0, :]).astype("int")
    return []

# 중복된 원 제거 함수
def remove_duplicate_circles(circles, distance_threshold):
    if len(circles) == 0:
        return []

    all_circles = [(x, y, r) for roi_circles in circles for (x, y, r) in roi_circles]
    unique_circles = []
    while len(all_circles) > 0:
        current_circle = all_circles.pop(0)
        unique_circles.append(current_circle)
        all_circles = [circle for circle in all_circles
                        if calculate_distance(current_circle, circle) > distance_threshold]
    
    return unique_circles

# 원 사이의 거리 계산 함수
def calculate_distance(circle1, circle2):
    x1, y1, _ = circle1
    x2, y2, _ = circle2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# 이미지에 텍스트 추가 함수
def Put_Text(src, text, pos, font_size, font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text(pos, text, font=font, fill=font_color)
    return np.array(img_pil)

# 이미지 처리 함수
def process_image(image_path, min_radius, max_radius, param1, param2, distance_threshold):
    image = load_image(image_path)
    if image is None:
        print("이미지를 불러오는 데 실패했습니다.")
        return
    
    # 면봉통 검출 및 마스킹 적용
    masked_image, mask = mask_outside_contour(image)

    # CLAHE 적용 (밝기 기반 처리)
    image_clahe = apply_clahe_based_on_brightness(masked_image)
    
    height, width = image_clahe.shape[:2]
    roi_size = (int(height * 5 / 17), int(width * 5 / 17))
    roi_list = create_roi(image_clahe, roi_size)
    
    all_circles = []
    preprocessed_images = []
    
    for roi, (x, y) in roi_list:
        pre = preprocess_roi(roi)
        preprocessed_images.append(pre)
        circles = detect_cotton_swabs(roi, pre, min_radius, max_radius, param1, param2, mask[y:y+roi.shape[0], x:x+roi.shape[1]])
        circles = [(cx + x, cy + y, r) for (cx, cy, r) in circles]
        all_circles.append(circles)
        cv2.rectangle(image_clahe, (x, y), (x + roi_size[1], y + roi_size[0]), (0, 255, 0), 2)

    unique_circles = remove_duplicate_circles(all_circles, distance_threshold)
    
    for (x, y, r) in unique_circles:
        cv2.circle(image_clahe, (x, y), r, (0, 0, 255), 2)
    
    total_cotton_swabs = len(unique_circles)
    print(f"Total cotton swabs detected: {total_cotton_swabs}")

    image_clahe = Put_Text(image_clahe, f"Count: {total_cotton_swabs}", (30, 60), 80, (0, 255, 255))
    
    max_width = 1000
    if image_clahe.shape[1] > max_width:
        scale_ratio = max_width / image_clahe.shape[1]
        new_height = int(image_clahe.shape[0] * scale_ratio)
        resized_image = cv2.resize(image_clahe, (max_width, new_height))
    else:
        resized_image = image_clahe

    cv2.imwrite("result.jpg", resized_image)
    
    plt.figure(figsize=(20, 10))
    plt.subplot(4, 5, 1)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Result Image (Total: {total_cotton_swabs})")
    plt.axis('off')
    
    for i, preprocessed_img in enumerate(preprocessed_images):
        row = i // 4
        col = i % 4
        plt.subplot(4, 5, 5 * row + col + 2)
        plt.imshow(preprocessed_img, cmap='gray')
        plt.title(f"ROI {i + 1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    cv2.imshow("Result", resized_image)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = select_image()
    min_radius = 25
    max_radius = 37
    param1 = 34
    param2 = 25
    distance_threshold = 40
    process_image(image_path, min_radius, max_radius, param1, param2, distance_threshold)

if __name__ == "__main__":
    main()
