# ◆ ESC : 창 닫기

import numpy as np
import cv2


def onChange():
    pass


def trackBar():
    def nothing(x):
        pass
    img = cv2.imread("bird.jpg", cv2.IMREAD_COLOR) 
    
    hor = 2000 # 변환될 가로 픽셀 사이즈
    ver = 2000 # 변환될 세로 픽셀 사이즈
    img = cv2.resize(img, (hor, ver)) 
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    cv2.namedWindow("color Track Bar")
    cv2.createTrackbar('H_low', 'color Track Bar', 50, 180, nothing)
    cv2.createTrackbar('H_high', 'color Track Bar', 180, 180, nothing)
    cv2.createTrackbar('S_low', 'color Track Bar', 100, 255, nothing)
    cv2.createTrackbar('S_high', 'color Track Bar', 255, 255, nothing)
    cv2.createTrackbar('V_low', 'color Track Bar', 0, 255, nothing)
    cv2.createTrackbar('V_high', 'color Track Bar', 255, 255, nothing)

    while True:

        # TRACKBAR
        H_low_val = cv2.getTrackbarPos('H_low', 'color Track Bar')
        H_high_val = cv2.getTrackbarPos('H_high', 'color Track Bar')
        S_low_val = cv2.getTrackbarPos('S_low', 'color Track Bar')
        S_high_val = cv2.getTrackbarPos('S_high', 'color Track Bar')
        V_low_val = cv2.getTrackbarPos('V_low', 'color Track Bar')
        V_high_val = cv2.getTrackbarPos('V_high', 'color Track Bar')

        #==============================
        # 컬러 영역의 from ~ to 
        lower = np.array([H_low_val, S_low_val, V_low_val]) # H / S / V
        upper = np.array([H_high_val, S_high_val, V_high_val])
        
        # 이미지에서 blue 영역
        mask = cv2.inRange(img_hsv, lower, upper)
        
        # bit 연산자를 통해 blue 영역만 남김
        res = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('color Track Bar', res)
        k = cv2.waitKey(1) & 0xFF

        if k == 27: # press esc
            break       
        #==============================

    cv2.destroyAllWindows()


trackBar()