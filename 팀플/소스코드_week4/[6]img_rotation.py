import cv2

img = cv2.imread("opencv_logo.png", cv2.IMREAD_COLOR)


# grab the dimensions of the image and calculate the center of the image
(h, w) = img.shape[:2]
(cX, cY) = (w / 2, h / 2)

# rotate our image by xx degrees
degree = 270
scale = 0.7 # 1.0보다 작은 값 --> 작아짐
            # 1.0보다 큰 값 --> 커짐
M = cv2.getRotationMatrix2D((cX, cY), degree, scale) 
img_rot = cv2.warpAffine(img, M, (w, h))


cv2.imshow("01.img", img) # 이미지 표시
cv2.imshow("02.rotation", img_rot)
cv2.waitKey(0) # 키를 누를 때까지 창을 띄워놓음
cv2.destroyAllWindows() # 키보드 자판의 아무 키나 누르면 창이 닫힘

