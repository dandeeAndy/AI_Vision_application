import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

# 이미지 불러오기 함수
def load_image(image_path):
    return cv2.imread(image_path)

# 면봉통 검출 함수 (가장 큰 원 찾기)
def detect_largest_circle(image, min_radius=120, max_radius=1200):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.3, minDist=50,
                                param1=100, param2=50, 
                                minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest_circle = max(circles, key=lambda x: x[2])  # 가장 큰 원
        return largest_circle
    return None

# 원 밖부분을 검은색으로 처리하는 함수
def mask_outside_circle(image, circle):
    masked_image = np.zeros_like(image)  # 같은 크기의 검은색 이미지 생성
    cx, cy, r = circle
    cv2.circle(masked_image, (cx, cy), r, (255, 255, 255), -1)  # 원 안 부분을 흰색으로
    masked_image = cv2.bitwise_and(image, masked_image)  # 원 안 부분만 남김
    return masked_image


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
    
    # V값 중 100에서 250 사이에서 가장 높은 빈도를 찾음
    v_peak_idx = np.argmax(hist[100:250]) + 100
    
    # 마스크 범위를 지정 (+30, -30)
    lower_v = max(v_peak_idx - 30, 0)
    upper_v = min(v_peak_idx + 30, 255)
    return lower_v, upper_v

#=========================================================================================================================

# ROI 전처리 함수
def preprocess_roi_1(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def preprocess_roi_2(roi):
    hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 각 ROI마다 V 채널 분석 후 마스크 적용
    h_channel, s_channel, v_channel = cv2.split(hsv_img)
    
    # V 채널의 히스토그램을 분석하여 마스크 범위 지정
    lower_v, upper_v = get_v_mask_range(v_channel)
    
    # HSV 범위를 설정 (H와 S는 신경쓰지 않고 V만 설정)
    lower_bound = np.array([0, 0, lower_v])
    upper_bound = np.array([180, 255, upper_v])
    
    # 마스크 적용
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    res = cv2.bitwise_and(roi, roi, mask=mask)
    
    gray_blurred = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    blurred_roi = cv2.GaussianBlur(gray_blurred, (5, 5), 0)
    
    # 3. 적응형 이진화 적용 (영역별 문턱값 처리)
    adaptive_thresh = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 4. 열림 연산 적용 (침식 후 팽창)
    kernel = np.ones((5, 5), np.uint8)  # 5x5 커널을 사용
    opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

    # 5. 소벨 연산자를 이용한 에지 검출 (수평과 수직 방향)
    sobel_x = cv2.Sobel(opened, cv2.CV_64F, 1, 0, ksize=3)  # 수평 방향 에지
    sobel_y = cv2.Sobel(opened, cv2.CV_64F, 0, 1, ksize=3)  # 수직 방향 에지

    # 절대값을 취하여 두 방향의 에지 결합
    sobel_combined = cv2.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    
    return sobel_combined  # 마스크가 적용되고 그레이스케일로 변환된 ROI 반환

#=========================================================================================================================

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
    
    # 중복 제거된 원들을 저장할 리스트
    unique_circles = []
    
    while len(circles) > 0:
        # 현재 원을 기준으로 설정
        current_circle = circles.pop(0)
        unique_circles.append(current_circle)
        
        # 현재 원과 가까운 원들을 제거
        circles = [circle for circle in circles
                    if calculate_distance(current_circle, circle) > distance_threshold]
    
    return unique_circles

# 원 사이의 거리 계산 함수
def calculate_distance(circle1, circle2):
    x1, y1, _ = circle1
    x2, y2, _ = circle2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# 이미지에 텍스트 추가 함수(한글)
def Put_Korean_Text(src, text, pos, font_size, font_color) :
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('C:/Users/schah/AppData/Local/Microsoft/Windows/Fonts/NanumSquareR.ttf', font_size)
    draw.text(pos, text, font=font, fill= font_color)
    return np.array(img_pil)

# 이미지 처리 함수
def process_image(image_path, min_radius, max_radius, param1, param2, distance_threshold):
    image = load_image(image_path)
    if image is None:
        print("이미지를 불러오는 데 실패했습니다.")
        return
    
    # 면봉통 검출
    largest_circle = detect_largest_circle(image)
    if largest_circle is None:
        print("면봉통을 찾을 수 없습니다.")
        return
    # 원 밖 부분을 검은색으로 마스크 처리
    masked_image = mask_outside_circle(image, largest_circle)
    
    # ROI 조각 생성
    height, width = masked_image.shape[:2]
    roi_size = (int(height * 5 / 17), int(width * 5 / 17))
    roi_list = create_roi(masked_image, roi_size)
    
    # 모든 ROI에서 검출된 원들을 저장할 리스트
    all_circles = []
    preprocessed_images = []
    
    for roi, (x, y) in roi_list:
        # 전처리 함수 중 가장 많은 원을 검출한 함수의 결과를 선택
        preprocessing_functions = [preprocess_roi_1, preprocess_roi_2]
        best_circles = []
        best_preprocessed = None
        max_circles = 0

        for preprocess_func in preprocessing_functions:
            preprocessed = preprocess_func(roi)
            circles = detect_cotton_swabs(roi, preprocessed, min_radius, max_radius, param1, param2)
        
            if len(circles) > max_circles:
                max_circles = len(circles)
                best_circles = circles
                best_preprocessed = preprocessed
        
        preprocessed_images.append(best_preprocessed)
        
        # 원의 좌표를 전체 이미지 기준으로 변환
        circles = [(cx + x, cy + y, r) for (cx, cy, r) in best_circles]
        all_circles.extend(circles)
        
        # ROI 영역 체크(초록색 사각형)
        cv2.rectangle(image, (x, y), (x + roi_size[1], y + roi_size[0]), (0, 255, 0), 2)
    
    # 중복 원 제거
    unique_circles = remove_duplicate_circles(all_circles, distance_threshold)
    
    # 결과 이미지에 원 그리기
    for (x, y, r) in unique_circles:
        cv2.circle(image, (x, y), r, (0, 0, 255), 2)
    
    total_cotton_swabs = len(unique_circles)
    print(f"검출된 총 면봉 개수: {total_cotton_swabs}")

    # 결과 이미지에 면봉 개수 출력
    image = Put_Korean_Text(image, f"면봉 수: {total_cotton_swabs}개", (30, 60), 80, (0, 255, 255))
    
    # 리사이즈 비율 설정 (1000 픽셀 이하로 조정)
    max_width = 1000
    if image.shape[1] > max_width:
        scale_ratio = max_width / image.shape[1]
        new_height = int(image.shape[0] * scale_ratio)
        resized_image = cv2.resize(image, (max_width, new_height))
    else:
        resized_image = image

    cv2.imwrite("result_1015.jpg", resized_image)
    
    # 결과 이미지와 전처리된 이미지 표시
    plt.figure(figsize=(20, 10))
    
    # 결과 이미지 표시
    plt.subplot(4, 5, 1)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Result Image (Total: {total_cotton_swabs})")
    plt.axis('off')
    
    # ROI 이미지 표시 (오른쪽으로 정렬)
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