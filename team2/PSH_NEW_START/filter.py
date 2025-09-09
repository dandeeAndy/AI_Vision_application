import cv2 as cv
import mediapipe as mp
import numpy as np
import glob
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
from collections import deque

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# 윈도우 설정
WINDOW_SIZE = (800, 800)
WEBCAM_SIZE = (640, 480)  # 웹캠 크기 설정

# 배경 이미지 로드
img_files = sorted(glob.glob('team2/OJH/lecture.png'))
if not img_files:
    print("No image files found")
    exit()
background = cv.imread(img_files[0])
background = cv.resize(background, WINDOW_SIZE)

# 각 방법별 레이어 초기화
layers = {
    'kalman': np.zeros((*WINDOW_SIZE, 3), dtype=np.uint8),
    'bezier': np.zeros((*WINDOW_SIZE, 3), dtype=np.uint8),
    'savgol': np.zeros((*WINDOW_SIZE, 3), dtype=np.uint8),
    'velocity': np.zeros((*WINDOW_SIZE, 3), dtype=np.uint8)
}

# Kalman 필터 초기화
def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,1,0],
                     [0,1,0,1],
                     [0,0,1,0],
                     [0,0,0,1]])
    kf.H = np.array([[1,0,0,0],
                     [0,1,0,0]])
    kf.R *= 0.1
    kf.Q *= 0.1
    return kf

# Bezier 곡선 계산
def bezier_smooth(points, num_points=50):
    if len(points) < 3:
        return points
    
    def get_bezier_points(p0, p1, p2, num):
        t = np.linspace(0, 1, num)
        b_points = np.zeros((num, 2))
        for i in range(num):
            b_points[i] = (1-t[i])**2 * p0 + 2*(1-t[i])*t[i] * p1 + t[i]**2 * p2
        return b_points.astype(np.int32)
    
    points = np.array(points)
    smoothed = []
    for i in range(len(points)-2):
        curve_points = get_bezier_points(points[i], points[i+1], points[i+2], num_points)
        smoothed.extend(curve_points)
    return smoothed

# 속도 기반 스무딩
def velocity_smooth(x, y, prev_points, max_velocity=50):
    if not prev_points:
        return x, y
    prev_x, prev_y = prev_points[-1]
    velocity = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
    
    if velocity > max_velocity:
        alpha = max_velocity / velocity
        smoothed_x = int(prev_x + (x - prev_x) * alpha)
        smoothed_y = int(prev_y + (y - prev_y) * alpha)
        return smoothed_x, smoothed_y
    return x, y

# 변수 초기화
kalman = init_kalman()
points_dict = {method: deque(maxlen=10) for method in layers.keys()}
prev_x = prev_y = None
drawing = False

# 카메라 초기화
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WEBCAM_SIZE[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, WEBCAM_SIZE[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv.flip(frame, 1)
    frame_display = frame.copy()  # 웹캠 표시용 복사본
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 웹캠 화면에 손 랜드마크 표시
            mp_drawing.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 검지 손가락 끝점
            index_finger = hand_landmarks.landmark[8]
            x = int(index_finger.x * WINDOW_SIZE[0])
            y = int(index_finger.y * WINDOW_SIZE[1])
            
            # 중지 손가락으로 그리기 모드 전환
            middle_finger = hand_landmarks.landmark[12]
            if middle_finger.y < hand_landmarks.landmark[10].y:  # 중지가 펴져있으면
                drawing = True
                # 그리기 모드일 때 웹캠에 상태 표시
                cv.putText(frame_display, "Drawing Mode", (10, 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                drawing = False
                prev_x = prev_y = None
                for points in points_dict.values():
                    points.clear()
                continue
            
            if drawing and prev_x is not None:
                # Kalman 필터
                kalman.predict()
                kalman.update(np.array([x, y]))
                smoothed_x, smoothed_y = kalman.x[0], kalman.x[1]
                cv.line(layers['kalman'], (int(prev_x), int(prev_y)), 
                       (int(smoothed_x), int(smoothed_y)), (0, 255, 0), 2)
                
                # Bezier 곡선
                points_dict['bezier'].append((x, y))
                if len(points_dict['bezier']) >= 3:
                    smoothed = bezier_smooth(list(points_dict['bezier']))
                    for i in range(len(smoothed)-1):
                        cv.line(layers['bezier'], tuple(smoothed[i]), 
                               tuple(smoothed[i+1]), (255, 0, 0), 2)
                
                # Savitzky-Golay
                points_dict['savgol'].append((x, y))
                if len(points_dict['savgol']) >= 5:
                    x_coords = np.array([p[0] for p in points_dict['savgol']])
                    y_coords = np.array([p[1] for p in points_dict['savgol']])
                    x_smooth = savgol_filter(x_coords, 5, 2)
                    y_smooth = savgol_filter(y_coords, 5, 2)
                    cv.line(layers['savgol'], 
                           (int(x_smooth[-2]), int(y_smooth[-2])),
                           (int(x_smooth[-1]), int(y_smooth[-1])), 
                           (0, 0, 255), 2)
                
                # 속도 기반
                smoothed_x, smoothed_y = velocity_smooth(x, y, list(points_dict['velocity']))
                points_dict['velocity'].append((smoothed_x, smoothed_y))
                if len(points_dict['velocity']) >= 2:
                    p1, p2 = list(points_dict['velocity'])[-2:]
                    cv.line(layers['velocity'], p1, p2, (255, 0, 255), 2)
            
            prev_x, prev_y = x, y
    
    # 결과 표시
    h, w = WINDOW_SIZE[1] // 2, WINDOW_SIZE[0] // 2
    top_left = cv.addWeighted(background[:h, :w], 0.7, layers['kalman'][:h, :w], 0.3, 0)
    top_right = cv.addWeighted(background[:h, w:], 0.7, layers['bezier'][:h, w:], 0.3, 0)
    bottom_left = cv.addWeighted(background[h:, :w], 0.7, layers['savgol'][h:, :w], 0.3, 0)
    bottom_right = cv.addWeighted(background[h:, w:], 0.7, layers['velocity'][h:, w:], 0.3, 0)
    
    # 텍스트 추가
    cv.putText(top_left, "Kalman", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(top_right, "Bezier", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.putText(bottom_left, "Savgol", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(bottom_right, "Velocity", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    # 화면 조합
    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))
    display = np.vstack((top, bottom))
    
    # 웹캠 창 표시
    cv.imshow('Webcam (Press ESC to quit)', frame_display)
    cv.imshow('Handwriting Smoothing Comparison', display)
    
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c'):  # Clear
        for layer in layers.values():
            layer.fill(0)

cap.release()
cv.destroyAllWindows()