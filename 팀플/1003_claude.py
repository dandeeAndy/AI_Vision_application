import cv2
import numpy as np

def load_image(image_path):
    return cv2.imread(image_path)

def create_roi(image, roi_size, overlap=0.8):
    height, width = image.shape[:2]
    roi_height, roi_width = roi_size
    
    roi_list = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            roi = image[y:y+roi_height, x:x+roi_width]
            roi_list.append((roi, (x, y)))
            x = int(x + roi_width * (1 - overlap))
        y = int(y + roi_height * (1 - overlap))
    
    return roi_list

def main():
    # 이미지 경로 설정
    image_path = "path_to_your_image.jpg"
    
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
    
    # 여기에 각 ROI에 대한 전처리 및 면봉 검출 코드를 추가할 예정입니다.

if __name__ == "__main__":
    main()