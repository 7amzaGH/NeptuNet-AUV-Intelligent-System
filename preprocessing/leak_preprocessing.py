import cv2
import numpy as np

def preprocess_leak(image):
    """
    Preprocessing for leak detection
    Returns binary mask with leak regions
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # Step 2: Dynamic thresholding (85th percentile)
    threshold_value = np.percentile(enhanced, 85)
    _, binary_mask = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 3: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=5)
    final_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Convert back to BGR for YOLO (expects 3 channels)
    final_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

    return final_mask_bgr
