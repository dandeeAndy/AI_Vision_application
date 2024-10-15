import os
import cv2
import numpy as np

# 이미지 리스트 불러오기
img_list = []
image_paths = ["mb_001.jpg", "mb_002.jpg", "mb_003.jpg", "mb_004.jpg", "mb_005.jpg",
                "mb_006.jpg", "mb_007.jpg", "mb_008.jpg", "mb_010.jpg", "mb_011.jpg"]
for img_path in image_paths:
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img_list.append(img)
        else:
            print(f"Warning: Unable to read image {img_path}")
    else:
        print(f"Warning: Image file {img_path} does not exist")

# 트랙바 콜백 함수 (필수지만 아무 동작도 필요 없음)
def nothing(x):
    pass

# 원 검출 및 트랙바로 실시간 조정하는 함수
def detect_circles_with_trackbar(img_list):
    # 윈도우 생성
    cv2.namedWindow("Trackbars")

    # 트랙바 생성
    cv2.createTrackbar("min_radius", "Trackbars", 30, 100, nothing)
    cv2.createTrackbar("max_radius", "Trackbars", 50, 200, nothing)
    cv2.createTrackbar("param1", "Trackbars", 40, 200, nothing)
    cv2.createTrackbar("param2", "Trackbars", 28, 100, nothing)
    cv2.createTrackbar("minDist", "Trackbars", 50, 200, nothing)

    while True:
        # 트랙바 값 읽기
        min_radius = cv2.getTrackbarPos("min_radius", "Trackbars")
        max_radius = cv2.getTrackbarPos("max_radius", "Trackbars")
        param1 = cv2.getTrackbarPos("param1", "Trackbars")
        param2 = cv2.getTrackbarPos("param2", "Trackbars")
        minDist = cv2.getTrackbarPos("minDist", "Trackbars")

        # 모든 이미지를 복사하여 원을 그린 후 보여줄 배열을 생성
        combined_image = []

        for image in img_list:
            # 이미지 그레이스케일 변환 및 블러 적용
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # HoughCircles로 원 검출
            circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.3, minDist=minDist,
                                        param1=param1, param2=param2, 
                                        minRadius=min_radius, maxRadius=max_radius)

            # 검출된 원을 그리기
            output_image = image.copy()
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
                    cv2.rectangle(output_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # 사이즈를 줄여서 보여줄 이미지를 추가
            resized_image = cv2.resize(output_image, (300, 300))
            combined_image.append(resized_image)

        # 9개의 이미지를 3x3 그리드로 결합하여 출력
        row1 = np.hstack(combined_image[0:3])
        row2 = np.hstack(combined_image[3:6])
        row3 = np.hstack(combined_image[6:9])
        grid_image = np.vstack((row1, row2, row3))

        # 결과 출력
        cv2.imshow("Detected Circles", grid_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    if len(img_list) > 0:
        detect_circles_with_trackbar(img_list)
    else:
        print("이미지가 없습니다.")

if __name__ == "__main__":
    main()
