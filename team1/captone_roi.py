import cv2
import numpy as np
import math
import pyzbar.pyzbar as pyzbar

SHORT_RANGE = (100, 200) 
LONG_RANGE = (100, 200)

def calculate_theta(rect):
    (cx, cy), (width, height), angle = rect
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    right_points = box[np.argsort(box[:, 0])[-2:]]
    right_mid_x = int(np.mean(right_points[:, 0]))
    right_mid_y = int(np.mean(right_points[:, 1]))
    vector_x = right_mid_x - cx
    vector_y = right_mid_y - cy
    theta = math.degrees(math.atan2(vector_y, vector_x))
    # 부호를 반대로 바꿉니다.
    if theta > 90:
        theta = theta - 180
    elif theta < -90:
        theta = theta + 180
    return round(theta)

def find_rectangle(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 180, 400)
    # 모폴로지 연산을 통해 엣지를 더 굵게 만듭니다.
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    max_area = 0
    
    for contour in contours:
        # 컨투어를 근사화합니다.
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 근사화된 컨투어가 4개의 꼭지점을 가지면 사각형으로 간주합니다.
        if len(approx) == 4:
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # 사각형의 넓이를 계산합니다.
            area = cv2.contourArea(box)
            
            # 사각형의 짧은 변과 긴 변 길이 계산
            width, height = rect[1]
            short_side = min(width, height)
            long_side = max(width, height)
            
            # 크기 제한 조건 확인 (조건을 조금 더 유연하게 조정)
            if (SHORT_RANGE[0] * 1 <= short_side <= SHORT_RANGE[1] * 1 and 
                LONG_RANGE[0] * 1 <= long_side <= LONG_RANGE[1] * 1):
                if area > max_area:
                    max_area = area
                    best_rect = rect
    
    if best_rect is not None:
        box = cv2.boxPoints(best_rect)
        box = np.int32(box)
        center = tuple(map(int, best_rect[0]))
        theta = calculate_theta(best_rect)  # 수정된 부분
        return box, center, theta, roi.copy()  # visualized_image 대신 roi.copy() 반환
    return None, None, None, None


def determine_floor(depth):
    """depth 값을 이용해 층 결정"""
    for i, (min_depth, max_depth) in enumerate(DEPTH_RANGES):
        if min_depth <= depth <= max_depth:
            return 3 - i  # 3층, 2층, 1층 순서
    return None

def process_roi(color_image, depth_image, roi, roi_index=None):
    """ROI 처리 및 QR 코드 검출"""
    color_roi = color_image[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
    depth_roi = depth_image[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
    
    decoded_objects = pyzbar.decode(color_roi)
    qr_data = None
    if decoded_objects:
        qr_data = decoded_objects[0].data.decode('utf-8')
    
    box, center, angle, visualized_roi = find_rectangle(color_roi)
    
    if box is not None and center is not None:
        global_center = (roi['x'] + center[0], roi['y'] + center[1])
        depth = depth_roi[center[1], center[0]]
        floor = determine_floor(depth)
        
        color_image[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']] = visualized_roi
        
        return qr_data, global_center, angle, box, floor, depth
    
    return None, None, None, None, None, None


box, center, angle, visualized_roi = find_rectangle(color_roi)