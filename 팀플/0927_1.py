import cv2
import numpy as np
import os

def load_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")
    return image

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, kernel_size=(9, 9), sigma=2):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def binary(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
           threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)

def opening(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def dilation(image, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

def erosion(image, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def detect_edges(image, ksize=3):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_combined = cv2.sqrt(sobel_x**2 + sobel_y**2)
    return cv2.convertScaleAbs(sobel_combined)

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_container(image):
    gray = grayscale(image)
    blurred = gaussian_blur(gray)
    
    # HoughCircles to detect the container
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=100, maxRadius=0)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        return largest_circle
    return None

def create_circular_mask(image, center, radius):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

# def process_image(image_path):
#     # Load the image
#     image = load_image(image_path)
#     original = image.copy()
    
#     # Detect the container
#     container = detect_container(image)
    
#     if container is not None:
#         # Create a circular ROI
#         center = (container[0], container[1])
#         radius = container[2]
#         mask = create_circular_mask(image, center, radius)
        
#         # Apply the mask to the image
#         roi = image.copy()
#         roi[~mask] = 0
        
#         # Draw the ROI circle on the original image
#         cv2.circle(original, center, radius, (255, 0, 0), 5)
        
#         # Process the ROI
#         gray_roi = grayscale(roi)
#         blurred_roi = gaussian_blur(gray_roi)
#         binary_roi = binary(blurred_roi)
#         opened_roi = opening(binary_roi)
#         closed_roi = closing(opened_roi)
#         edges_roi = detect_edges(closed_roi)
        
#         # Find contours
#         contours = find_contours(edges_roi)
        
#         # Draw contours on the original image
#         cv2.drawContours(original, contours, -1, (0, 255, 0), 2)
        
#         # Count the cotton swabs
#         cotton_swab_count = len(contours)
        
#         # Display results
#         cv2.putText(original, f"Cotton Swabs: {cotton_swab_count}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         # Show the image
#         cv2.imshow("Detected Cotton Swabs", original)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
#         return cotton_swab_count
#     else:
#         print("No container detected in the image.")
#         return None

# image_path = "mb_001.jpg"
# cotton_swab_count = process_image(image_path)
# if cotton_swab_count is not None:
#     print(f"Detected {cotton_swab_count} cotton swabs.")

def process_image(image_path):
    
    image = load_image(image_path)
    original = image.copy()
    
    container = detect_container(image)
    
    if container is not None:
        # Create a circular ROI
        center = (container[0], container[1])
        radius = container[2]
        mask = create_circular_mask(image, center, radius)
        
        # Apply the mask to the image
        roi = image.copy()
        roi[~mask] = 0
        
        # Draw the ROI circle on the original image
        cv2.circle(original, center, radius, (255, 0, 0), 5)
        
        # Process the ROI
        gray_roi = grayscale(roi)
        blurred_roi = gaussian_blur(gray_roi)
        binary_roi = binary(blurred_roi)
        opened_roi = opening(binary_roi)
        closed_roi = closing(opened_roi)
        edges_roi = detect_edges(closed_roi)
        
        # Find contours
        contours = find_contours(edges_roi)
        
        # Draw contours on the original image
        cv2.drawContours(original, contours, -1, (0, 255, 0), 2)
        
        # Count the cotton swabs
        cotton_swab_count = len(contours)
        
        # Display results on the image
        cv2.putText(original, f"Cotton Swabs: {cotton_swab_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return original, cotton_swab_count
    else:
        print(f"No container detected in the image: {image_path}")
        return None, 0

def display_results(images, cotton_swab_counts):
    rows = 2
    cols = 5
    scale_factor = 0.16
    
    resized_images = []
    for img in images:
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        resized_images.append(cv2.resize(img, (width, height)))
    
    min_height = min(image.shape[0] for image in resized_images)
    min_width = min(image.shape[1] for image in resized_images)
    
    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (min_width, min_height))
    
    combined_image = np.zeros((min_height * rows, min_width * cols, 3), dtype=np.uint8)
    
    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols
        combined_image[row * min_height:(row + 1) * min_height, col * min_width:(col + 1) * min_width] = img
    
    for i, count in enumerate(cotton_swab_counts):
        row = i // cols
        col = i % cols
        text_pos = (col * min_width + 10, (row + 1) * min_height - 10)
        cv2.putText(combined_image, f"Count: {count}", text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Cotton Swab Detection Results', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_paths = ["mb_001.jpg", "mb_002.jpg", "mb_003.jpg", "mb_004.jpg", "mb_005.jpg",
                   "mb_006.jpg", "mb_007.jpg", "mb_008.jpg", "mb_010.jpg", "mb_011.jpg"]
    
    processed_images = []
    cotton_swab_counts = []
    
    for image_path in image_paths:
        output, count = process_image(image_path)
        if output is not None:
            processed_images.append(output)
            cotton_swab_counts.append(count)
            print(f"Image: {image_path}, Cotton Swab Count: {count}")
    
    display_results(processed_images, cotton_swab_counts)