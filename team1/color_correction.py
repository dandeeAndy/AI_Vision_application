import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread("mb_001.jpg")

# 이미지 창 생성
cv2.namedWindow("Original Image")
cv2.imshow("Original Image", img)

# 선택한 영역 저장 변수
x1, y1, x2, y2 = 0, 0, 0, 0
selecting_roi = False

# 마우스 콜백 함수
def select_roi(event, x, y, flags, param):
    global x1, y1, x2, y2, selecting_roi, img_roi, img_corrected
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        selecting_roi = True
    elif event == cv2.EVENT_LBUTTONUP:
        x2, y2 = x, y
        selecting_roi = False
        img_roi = img[y1:y2, x1:x2]
        img_corrected = img_roi.copy()
        cv2.namedWindow("Color Correction")
        cv2.imshow("Color Correction - Original", img_roi)
        cv2.imshow("Color Correction - Corrected", img_corrected)

        cv2.createTrackbar("Brightness", "Color Correction", 50, 100, update_image)
        cv2.createTrackbar("Contrast", "Color Correction", 100, 200, update_image)
        cv2.createTrackbar("Red", "Color Correction", 0, 255, update_image)
        cv2.createTrackbar("Green", "Color Correction", 0, 255, update_image)
        cv2.createTrackbar("Blue", "Color Correction", 0, 255, update_image)
        cv2.createTrackbar("Hue", "Color Correction", 0, 179, update_image)
        cv2.createTrackbar("Saturation", "Color Correction", 0, 255, update_image)
        cv2.createTrackbar("Value", "Color Correction", 0, 255, update_image)

cv2.setMouseCallback("Original Image", select_roi)

# 트랙바 콜백 함수
def update_image(val):
    global img_corrected
    brightness = cv2.getTrackbarPos("Brightness", "Color Correction")
    contrast = cv2.getTrackbarPos("Contrast", "Color Correction")
    r = cv2.getTrackbarPos("Red", "Color Correction")
    g = cv2.getTrackbarPos("Green", "Color Correction")
    b = cv2.getTrackbarPos("Blue", "Color Correction")
    h = cv2.getTrackbarPos("Hue", "Color Correction")
    s = cv2.getTrackbarPos("Saturation", "Color Correction")
    v = cv2.getTrackbarPos("Value", "Color Correction")

    img_corrected = img_roi.copy()
    img_corrected = cv2.convertScaleAbs(img_corrected, alpha=contrast/100.0, beta=brightness-50)
    img_corrected = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2HSV)
    img_corrected[:, :, 0] = h
    img_corrected[:, :, 1] = s
    img_corrected[:, :, 2] = v
    img_corrected = cv2.cvtColor(img_corrected, cv2.COLOR_HSV2BGR)
    img_corrected[:, :, 2] = np.clip(img_corrected[:, :, 2] + b, 0, 255)
    img_corrected[:, :, 1] = np.clip(img_corrected[:, :, 1] + g, 0, 255)
    img_corrected[:, :, 0] = np.clip(img_corrected[:, :, 0] + r, 0, 255)
    cv2.imshow("Color Correction - Corrected", img_corrected)
    
# 이미지 저장
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite("corrected_image.jpg", img_corrected)
        print("Image saved as corrected_image.jpg")
        break