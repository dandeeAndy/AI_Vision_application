import cv2
import numpy as np

def load_image(image_path):
    return cv2.imread(image_path)

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
            
            # ROI가 이미지 경계를 벗어나지 않도록 조정
            y = min(y, height - roi_height)
            x = min(x, width - roi_width)
            
            roi = image[y:y+roi_height, x:x+roi_width]
            roi_list.append((roi, (x, y)))
    
    return roi_list

def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_cotton_swabs(roi, min_radius=10, max_radius=30):
    edges = preprocess_roi(roi)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return len(circles)
    return 0

def main():
    # 이미지 경로 설정
    image_path = "mb_001.jpg"
    
    # 이미지 불러오기
    image = load_image(image_path)
    
    if image is None:
        print("이미지를 불러오는 데 실패했습니다.")
        return
    
    # ROI 크기 계산
    height, width = image.shape[:2]
    roi_size = (int(height * 5/17), int(width * 5/17))
    
    # ROI 생성
    roi_list = create_roi(image, roi_size)
    
    print(f"생성된 ROI 개수: {len(roi_list)}")
    
    # 각 ROI에 대해 면봉 검출
    total_cotton_swabs = 0
    for roi, (x, y) in roi_list:
        cotton_swabs = detect_cotton_swabs(roi)
        total_cotton_swabs += cotton_swabs
    
    # ROI 시각화 (디버깅 목적)
    for idx, (roi, (x, y)) in enumerate(roi_list):
        cv2.rectangle(image, (x, y), (x + roi_size[1], y + roi_size[0]), (0, 255, 0), 2)
        cv2.putText(image, f"ROI {idx+1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    print(f"검출된 총 면봉 개수: {total_cotton_swabs}")
    
    # 결과 이미지 저장 (옵션)
    cv2.imwrite("result.jpg", image)

if __name__ == "__main__":
    main()