import cv2
import numpy as np

def apply_perspective_warp(image, bboxes, src_points=None, dst_points=None):
    """
    Applies a homography perspective warp to transform dashcam imagery to an 
    approximate top-down (birds-eye) perspective, adapting coordinates.
    
    bboxes: List of YOLO formatted boxes [[class_id, x_c, y_c, w, h], ...]
    """
    h_img, w_img = image.shape[:2]
    
    # Define default trapezoid (dashcam perspective view) if none provided
    if src_points is None:
        src_points = np.float32([
            [w_img * 0.15, h_img * 0.85], # Bottom-Left
            [w_img * 0.85, h_img * 0.85], # Bottom-Right
            [w_img * 0.60, h_img * 0.50], # Top-Right
            [w_img * 0.40, h_img * 0.50]  # Top-Left
        ])
    
    # Map to flat rectangle targets (top-down destination grid)
    if dst_points is None:
        dst_points = np.float32([
            [w_img * 0.10, h_img * 0.90],
            [w_img * 0.90, h_img * 0.90],
            [w_img * 0.90, h_img * 0.10],
            [w_img * 0.10, h_img * 0.10]
        ])
        
    # Compute the projection mapping matrices
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, M, (w_img, h_img))
    
    warped_bboxes = []
    for box in bboxes:
        cls_id, x_c, y_c, w, h = box
        
        # Convert YOLO normalized values back to absolute image pixel bounds
        abs_x1 = int((x_c - w / 2.0) * w_img)
        abs_y1 = int((y_c - h / 2.0) * h_img)
        abs_x2 = int((x_c + w / 2.0) * w_img)
        abs_y2 = int((y_c + h / 2.0) * h_img)
        
        # Define the 4 corners of the original box
        corners = np.array([
            [abs_x1, abs_y1], [abs_x2, abs_y1],
            [abs_x2, abs_y2], [abs_x1, abs_y2]
        ], dtype='float32').reshape(-1, 1, 2)
        
        # Warp the corner coordinate vectors
        warped_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
        
        # Construct a tight axis-aligned bounding box around the warped shape
        wx1 = np.min(warped_corners[:, 0])
        wy1 = np.min(warped_corners[:, 1])
        wx2 = np.max(warped_corners[:, 0])
        wy2 = np.max(warped_corners[:, 1])
        
        # Crop to visible frame limits
        wx1 = max(0, min(wx1, w_img - 1))
        wy1 = max(0, min(wy1, h_img - 1))
        wx2 = max(0, min(wx2, w_img - 1))
        wy2 = max(0, min(wy2, h_img - 1))
        
        # Recalculate back to normalized YOLO format
        nw = (wx2 - wx1) / float(w_img)
        nh = (wy2 - wy1) / float(h_img)
        nx_c = (wx1 + wx2) / 2.0 / float(w_img)
        ny_c = (wy1 + wy2) / 2.0 / float(h_img)
        
        if nw > 0.01 and nh > 0.01: # Discard squashed labels
            warped_bboxes.append([cls_id, nx_c, ny_c, nw, nh])
            
    return warped_image, warped_bboxes