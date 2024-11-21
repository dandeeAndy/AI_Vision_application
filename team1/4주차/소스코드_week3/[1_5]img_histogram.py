# 참고 링크 : http://www.gisdeveloper.co.kr/?p=6634

import cv2
from matplotlib import pyplot as plt # 히스토그램 출력을 위한 라이브러리 불러오기

# 이미지 열기
img = cv2.imread("bird.jpg", cv2.IMREAD_COLOR) 

# 히스토그램 표시
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 흑백으로 변환
plt.hist(img.ravel(), 256, [0,255]); 
plt.show()

# b / g / r 컬러 채널 별로 히스토그램 출력
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()















