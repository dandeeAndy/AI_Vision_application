import cv2 as cv
import mediapipe as mp
import glob
import numpy as np
import time

# (이전 초기화 코드는 동일하게 유지)

# 직선 그리기를 위한 변수 추가
start_point = None
temp_layer = None

def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True
    return False

cap = cv.VideoCapture(0)
while True:
    ret, frm = cap.read()
    if not ret:
        break
    frm = cv.flip(frm, 1)
    frm = cv.resize(frm, (800, 800))

    rgb = cv.cvtColor(frm, cv.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    slide_image_copy = slide_image.copy()
    pointer_layer = np.zeros_like(slide_image)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frm, i, mp_hands.HAND_CONNECTIONS)

            x, y = int(i.landmark[8].x * 800), int(i.landmark[8].y * 800)
            
            # (도구 선택 코드는 동일하게 유지)

            xi, yi = int(i.landmark[12].x * 800), int(i.landmark[12].y * 800)
            y9 = int(i.landmark[9].y * 800)

            # 수정된 draw 도구 구현
            if curr_tool == "draw":
                if index_raised(yi, y9):
                    if start_point is None:
                        # 선 그리기 시작점 설정
                        start_point = (x, y)
                        # 임시 레이어 생성
                        temp_layer = np.zeros_like(pen_layer)
                    else:
                        # 임시 레이어를 초기화하고 현재 드래그 중인 선을 표시
                        temp_layer = np.zeros_like(pen_layer)
                        cv.line(temp_layer, start_point, (x, y), (0, 0, 255), thick_pen)
                else:
                    # 손가락을 내리면 최종 선을 그림
                    if start_point is not None:
                        cv.line(pen_layer, start_point, (x, y), (0, 0, 255), thick_pen)
                        start_point = None
                        temp_layer = None

    # 레이어 합성 수정
    if temp_layer is not None:
        # 임시 레이어의 마스크 생성
        temp_gray = cv.cvtColor(temp_layer, cv.COLOR_BGR2GRAY)
        _, mask_temp = cv.threshold(temp_gray, 10, 255, cv.THRESH_BINARY)
        # 임시 레이어를 슬라이드에 합성
        cv.copyTo(src=temp_layer, dst=slide_image_copy, mask=mask_temp)

    # (이전의 레이어 합성 코드는 동일하게 유지)
    pen_gray = cv.cvtColor(pen_layer, cv.COLOR_BGR2GRAY)
    _, mask_pen = cv.threshold(pen_gray, 10, 255, cv.THRESH_BINARY)
    cv.copyTo(src=pen_layer, dst=slide_image_copy, mask=mask_pen)

    # (나머지 표시 및 키 처리 코드는 동일하게 유지)

cap.release()
cv.destroyAllWindows()