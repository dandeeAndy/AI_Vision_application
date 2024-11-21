#https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('bong44.jpg') # Read the test img

# [x, y]
src_1 = [20, 330]
src_2 = [1053, 332]
src_3 = [1165, 814]
src_4 = [63, 814]

dst_1 = [0, 0]
dst_2 = [1609, 0]
dst_3 = [1609, 814]
dst_4 = [0, 814]

src = np.float32([[src_1[0], src_1[1]], 
                 [src_2[0], src_2[1]], 
                 [src_3[0], src_3[1]], 
                 [src_4[0], src_4[1]]])

dst = np.float32([[dst_1[0], dst_1[1]], 
                 [dst_2[0], dst_2[1]], 
                 [dst_3[0], dst_3[1]], 
                 [dst_4[0], dst_4[1]]])

M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
warped_img = cv2.warpPerspective(img, M, (1609, 814)) # Image warping

#Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
#img_inv = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, IMAGE_H)) # Inverse transformation

red_color = (0, 0, 255)
img = cv2.circle(img, (src_1[0], src_1[1]), 10, red_color, -1)
img = cv2.circle(img, (src_2[0], src_2[1]), 10, red_color, -1)
img = cv2.circle(img, (src_3[0], src_3[1]), 10, red_color, -1)
img = cv2.circle(img, (src_4[0], src_4[1]), 10, red_color, -1)

img = cv2.line(img, (src_1[0], src_1[1]), (src_2[0], src_2[1]), red_color,1)
img = cv2.line(img, (src_2[0], src_2[1]), (src_3[0], src_3[1]), red_color,1)
img = cv2.line(img, (src_3[0], src_3[1]), (src_4[0], src_4[1]), red_color,1)
img = cv2.line(img, (src_4[0], src_4[1]), (src_1[0], src_1[1]), red_color,1)

cv2.imshow('img', img) # Show results
cv2.imshow('warped_img', warped_img) # Show results
cv2.waitKey(0)
cv2.destroyAllWindows()