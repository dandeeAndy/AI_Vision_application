import cv2
import numpy as np

# 이미지 파일 경로 설정
image_path = "mb_001.jpg"  # 경로 수정
output_image_path = "mb_001(save).jpg"  # 저장할 이미지 경로

# 이미지 열기 (컬러)
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 이미지가 제대로 로드됐는지 확인
if img is None:
    print("이미지를 불러올 수 없습니다.")
else:
    # 이미지 전처리: Gaussian Blur 적용
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 이진화 (Thresholding)
    _, binary = cv2.threshold(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), 130, 255, cv2.THRESH_BINARY)

    # 모폴로지 연산: 닫힘(closing) 연산
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Contour(윤곽선) 찾기
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("면봉 윤곽선을 찾을 수 없습니다.")
    else:
        # 면봉 크기(픽셀 수) 목록 생성
        areas = [cv2.contourArea(contour) for contour in contours]

        # 면봉 크기의 평균과 표준편차 계산
        mean_area = np.mean(areas)
        std_area = np.std(areas)

        # Z-점수로 특정 면봉의 크기가 이상치인지 확인하여 제거
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            z_score = (area - mean_area) / std_area if std_area > 0 else 0

            # 15보다 작고 700보다 큰 경우 제외
            if (area < 15 or area > 700) and abs(z_score) < 2:
                filtered_contours.append(contour)

        print(f'총 면봉 개수: {len(contours)}')
        print(f'이상치 제거 후 면봉 개수: {len(filtered_contours)}')

        # 겹치는 원 제거 및 원 표시
        centers = []
        unique_centers = []
        threshold_distance = 40  # 원 간의 최소 거리
        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:  # 분모가 0이 아닐 때만 계산
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))

                # 겹치는 원 확인
                if not any(np.linalg.norm(np.array((cX, cY)) - np.array(uc)) < threshold_distance for uc in unique_centers):
                    unique_centers.append((cX, cY))
                    cv2.circle(img, (cX, cY), 30, (0, 255, 0), 2)  # 초록색 원

        # 최종적으로 남은 면봉의 개수 출력
        print(f'최종 면봉 개수: {len(unique_centers)}')

    # 이진화된 이미지 확인 (선택 사항)
    binary_resized = cv2.resize(binary, (400, 300))
    cv2.imshow('Binary Image', binary_resized)

    # 최종 이미지 크기 조정
    img_resized = cv2.resize(img, (1000, 1000))
    cv2.imshow('Detected Cotton Swabs', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 이미지 저장
    cv2.imwrite(output_image_path, img)
    print(f"결과 이미지를 '{output_image_path}'에 저장했습니다.")