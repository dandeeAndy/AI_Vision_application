import os

import numpy as np
import cv2

# 이미지 읽기
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


# 모든 이미지를 동일한 크기로 리사이즈 (예: 500x500)
target_size = (500, 500)
resized_imgs = [cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) for img in img_list]

# HSV 변환
hsv_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in resized_imgs]

# 트랙바의 콜백 함수 (실제로는 필요 없음)
def nothing(x):
    pass

# 트랙바를 만들고 윈도우 생성
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Lower H', 'Image', 0, 180, nothing)
cv2.createTrackbar('Lower S', 'Image', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Image', 0, 255, nothing)
cv2.createTrackbar('Upper H', 'Image', 180, 180, nothing)
cv2.createTrackbar('Upper S', 'Image', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Image', 255, 255, nothing)

while True:
    # 트랙바에서 현재 값을 읽어옴
    lower_h = cv2.getTrackbarPos('Lower H', 'Image')
    lower_s = cv2.getTrackbarPos('Lower S', 'Image')
    lower_v = cv2.getTrackbarPos('Lower V', 'Image')
    upper_h = cv2.getTrackbarPos('Upper H', 'Image')
    upper_s = cv2.getTrackbarPos('Upper S', 'Image')
    upper_v = cv2.getTrackbarPos('Upper V', 'Image')

    # HSV 범위 설정
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # 결과 이미지를 담을 리스트
    result_imgs = []
    for img, hsv_img in zip(resized_imgs, hsv_imgs):
        # 색상 범위 마스크 생성
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        # 비트 연산을 통해 마스크 적용
        res = cv2.bitwise_and(img, img, mask=mask)
        result_imgs.append(res)

    # 이미지를 3x4 배열로 배치 (9장의 이미지를 3x4로 배치)
    rows = []
    for i in range(0, len(result_imgs), 4):  # 4개의 이미지를 한 행으로 묶음
        row_imgs = result_imgs[i:i+4]
        # 부족한 자리를 빈 이미지로 채움
        while len(row_imgs) < 4:
            empty_img = np.zeros_like(result_imgs[0])
            row_imgs.append(empty_img)
        rows.append(np.hstack(row_imgs))  # 가로로 붙이기

    # 각 행을 세로로 연결
    final_img = np.vstack(rows)

    # 이미지를 고정 비율로 리사이즈
    aspect_ratio = final_img.shape[1] / final_img.shape[0]
    new_width = 1200  # 원하는 너비
    new_height = int(new_width / aspect_ratio)  # 비율에 맞게 높이 계산
    res_resized = cv2.resize(final_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 결과 출력
    cv2.imshow('Image', res_resized)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 눌러 종료
        break

cv2.destroyAllWindows()
