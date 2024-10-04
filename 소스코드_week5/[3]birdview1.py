#https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lane.jpg') # Read the test img

src = np.float32([[270, 120], 
                  [395, 120], 
                  [560, 480], 
                  [110, 480]])

dst = np.float32([[0, 0], 
                  [640, 0], 
                  [640, 480], 
                  [0, 480]])

M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
warped_img = cv2.warpPerspective(img, M, (640, 480)) # Image warping

#Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
#img_inv = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, IMAGE_H)) # Inverse transformation

cv2.imshow('img', img) # Show results
cv2.imshow('warped_img', warped_img) # Show results
cv2.waitKey(0)
cv2.destroyAllWindows()