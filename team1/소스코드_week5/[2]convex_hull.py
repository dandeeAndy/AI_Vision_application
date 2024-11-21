# 참고 : https://m.blog.naver.com/handuelly/221803725224
import cv2

src = cv2.imread("bong44.jpg")
dst = src.copy()

hor = 1000 # 변환될 가로 픽셀 사이즈
ver = 1000 # 변환될 세로 픽셀 사이즈
src = cv2.resize(src, (hor, ver)) 
dst = cv2.resize(dst, (hor, ver)) 

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)


ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
# cv2.THRESH_BINARY_INV: 이진화 후 흑백반전 적용



contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#        · mode : 컨투어를 찾는 방법

                # cv2.RETR_EXTERNAL: 컨투어 라인 중 가장 바깥쪽의 라인만 찾음

                # cv2.RETR_LIST: 모든 컨투어 라인을 찾지만, 상하구조(hierachy)관계를 구성하지 않음

                # cv2.RETR_CCOMP: 모든 컨투어 라인을 찾고, 상하구조는 2 단계로 구성함

                # cv2.RETR_TREE: 모든 컨투어 라인을 찾고, 모든 상하구조를 구성함

#        · method : 컨투어를 찾을 때 사용하는 근사화 방법

                # cv2.CHAIN_APPROX_NONE: 모든 컨투어 포인트를 반환

                # cv2.CHAIN_APPROX_SIMPLE: 컨투어 라인을 그릴 수 있는 포인트만 반환

                # cv2.CHAIN_APPROX_TC89_L1: Teh_Chin 연결 근사 알고리즘 L1 버전을 적용해 컨투어 포인트
                
for i in contours:
    hull = cv2.convexHull(i, clockwise=True)
    cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2) 
    # 컨투어 그리기 (0, 0, 255): 선 색상, 2: 선 두께
    #print(hull)

cv2.imshow("binary", binary) # 이진화 결과
cv2.imshow("dst", dst) # 컨투어 표시 결과
cv2.waitKey(0)
cv2.destroyAllWindows()