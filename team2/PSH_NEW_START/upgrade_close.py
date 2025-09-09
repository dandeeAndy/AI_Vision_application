def process_frame(frame):
    # ... 기존 코드 ...

    if rotation_mode:
        if op.multi_hand_landmarks:
            for hand in op.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                if current_time - rotation_start_time > rotation_delay:
                    thumb_direction = detect_thumb_direction(hand, 4, 2)
                    if thumb_direction == "left":
                        rotation_angle -= 2
                    elif thumb_direction == "right":
                        rotation_angle += 2
                    
                    # 회전 모드에서의 실시간 미리보기
                    if rotation_mode and current_cropped_image is not None:
                        rotated_image = rotate_image(current_cropped_image, rotation_angle)
                        cv.imshow("Rotating Image", rotated_image)

                    # 모든 손가락이 펴져있을 때 (회전 확정)
                    if is_all_fingers_up(hand):
                        rotated_image = rotate_image(current_cropped_image, rotation_angle)
                        if len(summary_images) >= 4:
                            summary_images.pop(0)
                        summary_images.append(rotated_image.copy())
                        cv.destroyWindow("Rotating Image")  # 회전 창 즉시 닫기
                        cv.destroyWindow("Cropped Image")   # 크롭 창 닫기
                        cv.destroyWindow("Additional Materials")  # 추가 창 닫기
                        rotation_mode = False
                        curr_tool = 'laser pointer'
                        print("Rotation confirmed. Switched back to laser pointer.")
                        return pointer_layer  # 즉시 반환

    # ... 기존 코드 ...

def process_crop_tool(hand, x, y, fx=1.0, fy=1.0, tool_type="crop"):
    # ... 기존 코드 ...
    
    # 도구 변경 시 이전 창들 닫기
    if curr_tool != tool_type:
        cv.destroyWindow("Rotating Image")
        cv.destroyWindow("Cropped Image")
        cv.destroyWindow("Additional Materials")

    # ... 나머지 코드 ...