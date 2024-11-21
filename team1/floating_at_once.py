import cv2
import numpy as np
import at_once as ASC

mbimg = ["mb_001.jpg", "mb_002.jpg", "mb_003.jpg", "mb_004.jpg", "mb_005.jpg", "mb_006.jpg", "mb_007.jpg", 
         "mb_008.jpg", "mb_009.jpg", "mb_010.jpg", "mb_011.jpg", "mb_025.jpg", "mb_028.jpg", "mb_031.jpg"]

# Ctrl + 1 (주석해제)로 사용
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
    
#     # 선택하여 사용↓↓↓
#     cv2.imshow(f'OI {i+1}', resized_mbimg) #원본 이미지
#     # cv2.imshow(f'GI {i+1}', resized_image) #그레이스케일 이미지


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
    
    process_img = ASC.process_image(filename, 43)
    width = int(process_img.shape[1] * scale_factor)
    height = int(process_img.shape[0] * scale_factor)
    resized_image = cv2.resize(process_img, (width, height))
    
    
    images.append(resized_image)

min_height = min(image.shape[0] for image in images)
min_width = min(image.shape[1] for image in images)

for i in range(len(images)):
    images[i] = cv2.resize(images[i], (min_width, min_height))

rows = 2
cols = 7
combined_image = np.zeros((min_height * rows, min_width * cols, 3), dtype=np.uint8)

for i in range(len(images)):
    row = i // cols
    col = i % cols
    combined_image[row * min_height:(row + 1) * min_height, col * min_width:(col + 1) * min_width] = images[i]

cv2.imshow('Combined Images', combined_image)
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 창 닫기