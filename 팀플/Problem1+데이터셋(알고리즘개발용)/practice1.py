# 참고링크 : https://076923.github.io/posts/Python-opencv-12/

import cv2
import numpy as np

mbimg = ["mb_001.jpg", "mb_002.jpg", "mb_003.jpg", "mb_004.jpg", "mb_005.jpg", 
          "mb_006.jpg", "mb_007.jpg", "mb_008.jpg", "mb_010.jpg", "mb_011.jpg"]

#=========================================================================================
#                                ↓한 번에 10개 띄우는 코드(10개 창)↓
#=========================================================================================

# img_gray = []

# scale_factor = 0.1
# resized_images = []

# for i, filename in enumerate(mbimg):
#     color_image = cv2.imread(filename, cv2.IMREAD_COLOR)
#     if color_image is None:
#         print(f"Error loading image {filename}. Check the file path.")
#         continue
    
#     gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#     img_gray.append(gray_image)

#     width_color = int(color_image.shape[1] * scale_factor)
#     height_color = int(color_image.shape[0] * scale_factor)
#     resized_mbimg = cv2.resize(color_image, (width_color, height_color))

#     width_gray = int(gray_image.shape[1] * scale_factor)
#     height_gray = int(gray_image.shape[0] * scale_factor)
#     resized_image = cv2.resize(gray_image, (width_gray, height_gray))
    
#     # cv2.imshow(f'OI {i+1}', resized_mbimg) #원본 이미지
#     cv2.imshow(f'GI {i+1}', resized_image) #그레이스케일 이미지


# cv2.waitKey(0)
# cv2.destroyAllWindows()



#=========================================================================================
#                                ↓한 번에 10개 띄우는 코드(1개 창)↓
#=========================================================================================

scale_factor = 0.14

images = []
for filename in mbimg:
    color_image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if color_image is None:
        print(f"Error loading image {filename}. Check the file path.")
        continue
    
    width = int(color_image.shape[1] * scale_factor)
    height = int(color_image.shape[0] * scale_factor)
    resized_image = cv2.resize(color_image, (width, height))
    
    images.append(resized_image)

min_height = min(image.shape[0] for image in images)
min_width = min(image.shape[1] for image in images)

for i in range(len(images)):
    images[i] = cv2.resize(images[i], (min_width, min_height))

rows = 2
cols = 5
combined_image = np.zeros((min_height * rows, min_width * cols, 3), dtype=np.uint8)

for i in range(len(images)):
    row = i // cols
    col = i % cols
    combined_image[row * min_height:(row + 1) * min_height, col * min_width:(col + 1) * min_width] = images[i]

cv2.imshow('Combined Images', combined_image)
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 창 닫기




# #=============================================
# # 이미지 이진화
# threhold = 100 # 이 값보다 픽셀값이 큰 영역은 흰색(255), 작으면 검정색(0)으로 변환
# ret, img_binary = cv2.threshold(img_gray, threhold, 255, cv2.THRESH_BINARY)
# # 임계값(threshold)을 0에서 255까지 다양하게 변화시키면서 실행해보세요.
# # 이진화를 적용하면 이미지는 무조건 흰색(255값) 또는 검정색(0값)만 포함합니다.
# # 변수창에서 img_binary를 더블 클릭하여 픽셀값을 직접 확인해보세요.
# #=============================================

# # 이미지 표시
# cv2.imshow('before', img_gray) # 이진화 적용 전
# cv2.resizeWindow('before', 400, 300)


# scale_factor = 0.2

# width = int(img_gray.shape[1] * scale_factor)
# height = int(img_gray.shape[0] * scale_factor)
# resized_image = cv2.resize(img_gray, (width, height))

# cv2.imshow('Resized Image', resized_image)

# # cv2.imshow('after', img_binary) # 이진화 적용 후
# cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
# cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘
