import cv2
import numpy as np
import random

def apply_perspective_warp(image, bboxes, distortion_scale=0.06):
    """
    Applies a subtle, highly stable perspective shift. Uses a safe constant 
    asphalt gray border to completely eliminate mirror reflection artifacts.
    """
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

    # Low distortion limits prevent aggressive scaling out
    dx = w * distortion_scale
    dy = h * distortion_scale

    dst_points = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],
        [w - 1 - random.uniform(0, dx), random.uniform(0, dy)],
        [w - 1 - random.uniform(0, dx), h - 1 - random.uniform(0, dy)],
        [random.uniform(0, dx), h - 1 - random.uniform(0, dy)]
    ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Using a neutral gray constant border color (115, 115, 115) to emulate road surface
    warped_image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(115, 115, 115))

    warped_bboxes = []
    for box in bboxes:
        cls_id, x_c, y_c, box_w, box_h = box
        x1 = int((x_c - box_w / 2.0) * w)
        y1 = int((y_c - box_h / 2.0) * h)
        x2 = int((x_c + box_w / 2.0) * w)
        y2 = int((y_c + box_h / 2.0) * h)

        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='float32').reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

        wx1, wy1 = np.min(warped_corners[:, 0]), np.min(warped_corners[:, 1])
        wx2, wy2 = np.max(warped_corners[:, 0]), np.max(warped_corners[:, 1])

        wx1 = max(0, min(wx1, w - 1))
        wy1 = max(0, min(wy1, h - 1))
        wx2 = max(0, min(wx2, w - 1))
        wy2 = max(0, min(wy2, h - 1))

        nw = (wx2 - wx1) / float(w)
        nh = (wy2 - wy1) / float(h)
        nx_c = (wx1 + wx2) / 2.0 / float(w)
        ny_c = (wy1 + wy2) / 2.0 / float(h)

        if nw > 0.01 and nh > 0.01:
            warped_bboxes.append([cls_id, nx_c, ny_c, nw, nh])

    if warped_bboxes:
        warped_image, warped_bboxes = apply_target_crop(warped_image, warped_bboxes)

    return warped_image, warped_bboxes


def apply_target_crop(image, bboxes, crop_scale=0.78):
    """Crops around the main labeled box without changing perspective."""
    h, w = image.shape[:2]

    anchor_box = max(bboxes, key=lambda box: box[3] * box[4])
    center_x = float(anchor_box[1] * w)
    center_y = float(anchor_box[2] * h)

    crop_w = max(1, int(w * crop_scale))
    crop_h = max(1, int(h * crop_scale))

    crop_x1 = max(0, min(int(center_x - crop_w / 2), w - crop_w))
    crop_y1 = max(0, min(int(center_y - crop_h / 2), h - crop_h))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    resized_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)

    cropped_bboxes = []
    for box in bboxes:
        cls_id, x_c, y_c, box_w, box_h = box
        abs_x1 = int((x_c - box_w / 2.0) * w)
        abs_y1 = int((y_c - box_h / 2.0) * h)
        abs_x2 = int((x_c + box_w / 2.0) * w)
        abs_y2 = int((y_c + box_h / 2.0) * h)

        ix1 = max(abs_x1, crop_x1)
        iy1 = max(abs_y1, crop_y1)
        ix2 = min(abs_x2, crop_x2)
        iy2 = min(abs_y2, crop_y2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        nx1 = (ix1 - crop_x1) / float(crop_w)
        ny1 = (iy1 - crop_y1) / float(crop_h)
        nx2 = (ix2 - crop_x1) / float(crop_w)
        ny2 = (iy2 - crop_y1) / float(crop_h)

        nx_c = (nx1 + nx2) / 2.0
        ny_c = (ny1 + ny2) / 2.0
        nw = nx2 - nx1
        nh = ny2 - ny1

        if nw > 0.01 and nh > 0.01:
            cropped_bboxes.append([cls_id, nx_c, ny_c, nw, nh])

    return resized_image, cropped_bboxes

def apply_motion_blur(image, kernel_size=9):
    """Pronounced linear velocity blur filter."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    return cv2.filter2D(image, -1, kernel / kernel_size)

def is_image_blurred(image, threshold=120.0):
    """Detects blur using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def apply_night_effect(image):
    """Simulates dark environment exposure drop."""
    night_img = image.astype(np.float32) * 0.40
    night_img[:, :, 1] += 8
    night_img[:, :, 2] += 18
    return np.clip(night_img, 0, 255).astype(np.uint8)

def apply_lens_flare(image):
    """
    Simulates a soft, cinematic light leak creeping in from the edges 
    by heavily diffusing massive glowing gradients across the image plane.
    """
    h, w = image.shape[:2]
    flare_layer = np.zeros_like(image, dtype=np.uint8)

    # Randomly pick which edge the light leaks from (0: Top, 1: Left, 2: Right)
    edge = random.choice([0, 1, 2])

    # Choose a warm/retro light leak color profile (BGR: Orange/Pinkish hue)
    leak_color = (random.randint(180, 220), random.randint(200, 235), 255)

    if edge == 0:  # Leaks down from the top edge
        cx = random.randint(0, w)
        cy = 0
        radius = random.randint(int(h * 0.4), int(h * 0.7))
        cv2.circle(flare_layer, (cx, cy), radius, leak_color, -1)
    elif edge == 1:  # Leaks from the left edge
        cx = 0
        cy = random.randint(0, h)
        radius = random.randint(int(w * 0.4), int(w * 0.7))
        cv2.circle(flare_layer, (cx, cy), radius, leak_color, -1)
    else:  # Leaks from the right edge
        cx = w
        cy = random.randint(0, h)
        radius = random.randint(int(w * 0.4), int(w * 0.7))
        cv2.circle(flare_layer, (cx, cy), radius, leak_color, -1)

    # Apply a massive blur to completely wash out the circular shapes into a smooth gradient
    # The kernel must be odd (e.g., 151x151)
    flare_layer = cv2.GaussianBlur(flare_layer, (151, 151), 0)

    # Screen blend/Linear add onto the original image
    # 0.45 weight gives a strong, hazy washout effect over the asphalt
    return cv2.addWeighted(image, 1.0, flare_layer, 0.45, 0)

def apply_random_degradation(image):
    """Evaluates frame brightness. Skip night effects if already dark."""
    # Compute average grayscale pixel value
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    effects = ["night", "blur", "flare", "clean"]
    weights = [0.40, 0.20, 0.20, 0.20]

    chosen_effect = random.choices(effects, weights=weights, k=1)[0]

    # Gatekeeper check: If image is already dark (mean value < 85), route away from night
    if chosen_effect == "night" and avg_brightness < 85:
        chosen_effect = random.choice(["flare", "clean"])

    if chosen_effect == "night":
        return apply_night_effect(image), "Night Shift"
    elif chosen_effect == "blur":
        if is_image_blurred(image):
            return image, "Clean Baseline"
        return apply_motion_blur(image), "Motion Blur"
    elif chosen_effect == "flare":
        return apply_lens_flare(image), "Lens Flare"
    else:
        return image, "Clean Baseline"
