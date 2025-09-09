if summary_images:
    slide_display = cv.resize(slide_image_copy, (800, 600))
    summary_height = 200
    summary_width = 800
    # 흰색 배경으로 초기화 (255로 설정)
    summary_area = np.full((summary_height, summary_width, 3), 255, dtype=np.uint8)
    section_width = summary_width // 4

    # 검은색 구분선 그리기
    separator = np.zeros((2, summary_width, 3), dtype=np.uint8)  # 2픽셀 두께의 검은색 선
    
    for idx, (img, size) in enumerate(zip(summary_images[-4:], image_sizes[-4:])):
        if img is None:
            continue
        h, w = size
        
        # 다른 이미지들과 크기 비교
        max_h = max(s[0] for s in image_sizes)
        max_w = max(s[1] for s in image_sizes)
        
        # 현재 이미지가 가장 큰 경우
        if h >= max_h or w >= max_w:
            scale = summary_height / h
            new_h = summary_height
            new_w = int(w * scale)
            
            if new_w > section_width:
                scale = section_width / w
                new_w = section_width
                new_h = int(h * scale)
        else:
            scale = min(section_width/max_w, summary_height/max_h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        
        resized_img = cv.resize(img, (new_w, new_h))
        x_offset = idx * section_width + (section_width - new_w) // 2
        y_offset = (summary_height - new_h) // 2
        summary_area[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    # 슬라이드 디스플레이와 구분선, 요약 영역을 합치기
    final_display = np.vstack([slide_display, separator, summary_area])
    cv.imshow("Lecture", final_display)