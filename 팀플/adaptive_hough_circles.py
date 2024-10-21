
def adaptive_hough_circles(image, min_radius_ratio=0.01, max_radius_ratio=0.02, dp_scale=0.00015):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    min_dimension = min(height, width)
    
    dp = 1 + dp_scale * min_dimension
    
    min_dist = int(0.03 * min_dimension)
    
    min_radius = int(min_radius_ratio * min_dimension)
    max_radius = int(max_radius_ratio * min_dimension)
    
    # Hough Circles 적용
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=dp,minDist=min_dist,
                                param1=50,param2=30,minRadius=min_radius,maxRadius=max_radius)
    
    return circles
