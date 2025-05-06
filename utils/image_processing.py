import cv2
import numpy as np
# Removed matplotlib import as plotting happens on frontend

def merge_boxes(boxes, overlapThresh=0.2):
    """Merges overlapping bounding boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Using x2 for sorting might be slightly better for left-to-right merging tendency
    idxs = np.argsort(x2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last] # Indices to remove

        # Loop through remaining indices
        for pos in range(last):
            j = idxs[pos]

            # Find intersection coordinates
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # Compute width and height of intersection
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # Compute overlap ratio (intersection area / area of the current box)
            overlap = float(w * h) / area[j]

            # If overlap is greater than threshold, suppress the current box
            if overlap > overlapThresh:
                suppress.append(pos)

        # Delete suppressed indices
        idxs = np.delete(idxs, suppress)

    # Return only the picked boxes
    return boxes[pick].astype("int").tolist() # Return as list of lists

def get_plant_bounding_boxes(image_path, show=True):
    """
    Detects green regions in an image and returns merged bounding boxes.

    Args:
        image_path (str): Path to the input image.
        show (bool): Flag to control plotting (unused in web app backend).

    Returns:
        list: A list of bounding boxes [[x1, y1, x2, y2], ...].
              Returns empty list if image not found or no boxes detected.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return [] # Return empty list on error

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Color Thresholding for green areas
    # Adjusted green range slightly for potentially varying lighting
    lower_green = np.array([30, 35, 35])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 2. Morphological operations to clean up the mask
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Close small gaps within plant regions
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_ellipse, iterations=3)
    # Remove small noise pixels (opening = erosion followed by dilation)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_rect, iterations=2)

    # 3. Find contours on the cleaned mask
    contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Initial bounding boxes from contours (with area filtering)
    initial_boxes = []
    min_area = 250 # Increased minimum area slightly
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            initial_boxes.append([x, y, x + w, y + h]) # Store as [x1, y1, x2, y2]

    if not initial_boxes:
        print("No initial green regions detected.")
        return []

    # 5. Merge overlapping boxes
    merged_boxes = merge_boxes(initial_boxes, overlapThresh=0.1) # Lowered threshold slightly

    # 6. Return array (list of lists)
    print(f"Detected {len(merged_boxes)} final boxes.")
    return merged_boxes


def get_cropped_patches(image_path, boxes, scale_up=False, scale_factor=1, show=True):
    """
    Extracts and enhances cropped patches from an image based on bounding boxes.

    Args:
        image_path (str): Path to the input image.
        boxes (list): List of bounding boxes [[x1, y1, x2, y2], ...].
        scale_up (bool): Whether to scale up the cropped patches.
        scale_factor (int): Factor by which to scale up (if scale_up is True).
        show (bool): Flag to control plotting (unused in web app backend).

    Returns:
        list: A list of processed cropped image patches (as numpy arrays).
              Returns empty list if image not found.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at '{image_path}' not found.")
        return []

    cropped_images_data = []

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Ensure coordinates are within image bounds
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        x1, x2 = max(0, x1), min(image.shape[1], x2)

        # Crop the image
        crop = image[y1:y2, x1:x2]

        # Check if crop is valid
        if crop.size == 0:
            print(f"Warning: Skipping empty crop for box {i+1} at [{x1},{y1},{x2},{y2}]")
            continue

        h, w = crop.shape[:2]

        # Optional scaling
        if scale_up and scale_factor > 1:
            crop = cv2.resize(crop, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

        # --- Enhancement Steps ---
        # Convert to LAB color space for contrast enhancement
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            enhanced_crop = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

            # Unsharp masking for sharpening (apply gently)
            blurred = cv2.GaussianBlur(enhanced_crop, (0, 0), sigmaX=1.0) # Reduced sigma slightly
            sharpened = cv2.addWeighted(enhanced_crop, 1.3, blurred, -0.3, 0) # Adjusted weights

            cropped_images_data.append(sharpened)

        except cv2.error as e:
            print(f"Warning: OpenCV error processing patch {i+1}: {e}. Skipping patch.")
            # Optionally append the original crop or None if processing fails
            # cropped_images_data.append(crop) # Or append None
            continue

    return cropped_images_data


def classify_plant_health_rules(cropped_images, green_threshold=0.40, black_pixel_ratio_threshold=0.15,
                          hole_defect_threshold=0.015, brightness_threshold=75):
    """
    Classifies cropped plant images as healthy or not healthy based on rules.
    (This is the rule-based function from the notebook, not ML-based)

    Args:
        cropped_images (list of np.ndarray): List of cropped plant images (BGR format).
        green_threshold (float): Minimum green dominance ratio.
        black_pixel_ratio_threshold (float): Max allowed black area ratio.
        hole_defect_threshold (float): Max allowed proportion of edge/holes.
        brightness_threshold (int): Minimum average brightness to consider healthy.

    Returns:
        List of tuples: (image_index, 'healthy' or 'not healthy')
    """
    results = []

    for idx, img in enumerate(cropped_images):
        if img is None or img.size == 0:
            print(f"Warning: Skipping invalid image at index {idx}")
            results.append((idx + 1, 'error - invalid image'))
            continue

        h, w = img.shape[:2]
        if h == 0 or w == 0:
             print(f"Warning: Skipping zero-dimension image at index {idx}")
             results.append((idx + 1, 'error - zero dimension'))
             continue

        # --- Calculations (handle potential division by zero) ---
        total_pixels = float(h * w)
        if total_pixels == 0:
            results.append((idx + 1, 'error - zero area'))
            continue

        # 1. Green dominance check
        try:
            b, g, r = cv2.split(img)
            # Consider a pixel green if G is highest and above a minimum intensity
            green_pixels_mask = (g > r) & (g > b) & (g > 50)
            green_pixels_count = np.sum(green_pixels_mask)
            green_ratio = green_pixels_count / total_pixels
        except Exception as e:
            print(f"Error in green check for patch {idx+1}: {e}")
            green_ratio = 0 # Assume not green on error

        # 2. Brightness check
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
        except Exception as e:
             print(f"Error in brightness check for patch {idx+1}: {e}")
             avg_brightness = 0 # Assume dark on error

        # 3. Black spot detection
        try:
            # Use a lower threshold for black
            black_pixels_count = np.sum(gray < 30)
            black_ratio = black_pixels_count / total_pixels
        except Exception as e:
            print(f"Error in black spot check for patch {idx+1}: {e}")
            black_ratio = 1.0 # Assume problematic on error

        # 4. Leaf holes or tears detection (simplified)
        try:
            # Use the green mask to avoid detecting background as holes
            # Invert the green mask (non-green areas within the patch)
            non_green_mask = cv2.bitwise_not(green_pixels_mask.astype(np.uint8) * 255)

            # Find contours on the non-green mask
            contours, _ = cv2.findContours(non_green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # RETR_LIST finds inner contours

            # Filter contours that are likely holes (not too small, not too large)
            min_hole_area = 15 # Adjust based on expected hole size relative to patch
            max_hole_area = total_pixels * 0.1 # Max 10% of patch area
            hole_like_area = sum([cv2.contourArea(cnt) for cnt in contours if min_hole_area < cv2.contourArea(cnt) < max_hole_area])
            hole_ratio = hole_like_area / total_pixels
        except Exception as e:
            print(f"Error in hole check for patch {idx+1}: {e}")
            hole_ratio = 1.0 # Assume problematic on error


        # --- Decision Logic ---
        is_unhealthy = False
        reasons = []
        if green_ratio < green_threshold:
            is_unhealthy = True
            reasons.append(f"Low green ratio ({green_ratio:.2f} < {green_threshold})")
        if black_ratio > black_pixel_ratio_threshold:
            is_unhealthy = True
            reasons.append(f"High black ratio ({black_ratio:.2f} > {black_pixel_ratio_threshold})")
        if hole_ratio > hole_defect_threshold:
            is_unhealthy = True
            reasons.append(f"High hole ratio ({hole_ratio:.3f} > {hole_defect_threshold})")
        if avg_brightness < brightness_threshold:
             is_unhealthy = True
             reasons.append(f"Low brightness ({avg_brightness:.1f} < {brightness_threshold})")

        # Assign label
        label = 'not healthy' if is_unhealthy else 'healthy'
        print(f"Patch {idx+1}: Label={label}. Reasons: {', '.join(reasons) if reasons else 'None'}") # Logging

        results.append((idx + 1, label))

    return results
