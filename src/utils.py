"""
Utility functions for phone usage detection
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
import config

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_phone_center(bbox):
    """Get center point of phone bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def get_hand_center(hand_landmarks):
    """Get center point of hand using palm landmarks"""
    # Use palm center (landmark 9) as hand center
    palm_center = hand_landmarks.landmark[9]
    return (palm_center.x, palm_center.y)

def is_phone_hand_close(phone_center, hand_center, frame_shape):
    """Check if phone and hand are close enough for active usage"""
    # Phone center is already in pixel coordinates
    # Hand center is in normalized coordinates (0-1), need to convert to pixels
    h, w = frame_shape[:2]
    phone_pixel = phone_center  # Already in pixels
    hand_pixel = (hand_center[0] * w, hand_center[1] * h)  # Convert normalized to pixels
    
    distance = calculate_distance(phone_pixel, hand_pixel)
    return distance <= config.PHONE_HAND_DISTANCE_THRESHOLD

def detect_motion(phone_positions, frame_idx):
    """Detect if phone is moving based on position history"""
    if len(phone_positions) < config.MIN_MOTION_FRAMES:
        return False
    
    # Calculate average movement over recent frames
    recent_positions = phone_positions[-config.MIN_MOTION_FRAMES:]
    total_movement = 0
    
    for i in range(1, len(recent_positions)):
        if recent_positions[i] and recent_positions[i-1]:
            movement = calculate_distance(recent_positions[i], recent_positions[i-1])
            total_movement += movement
    
    avg_movement = total_movement / (len(recent_positions) - 1)
    return avg_movement > config.MOTION_THRESHOLD

def create_output_directory():
    """Create output directory if it doesn't exist"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    return config.OUTPUT_DIR

def save_report(usage_data, video_path, output_path):
    """Save phone usage report to JSON file"""
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Convert usage data to JSON-serializable format
    serializable_usage_data = convert_numpy_types(usage_data)
    
    report = {
        "video_path": video_path,
        "output_path": output_path,
        "processing_time": datetime.now().isoformat(),
        "total_frames": len(usage_data),
        "active_phone_usage_frames": sum(1 for frame in usage_data if frame.get("active_phone_usage", False)),
        "tap_to_pay_usage_frames": sum(1 for frame in usage_data if frame.get("tap_to_pay_usage", False)),
        "phone_usage_percentage": float((sum(1 for frame in usage_data if frame.get("active_phone_usage", False)) / len(usage_data)) * 100),
        "frame_details": serializable_usage_data
    }
    
    report_path = os.path.join(config.OUTPUT_DIR, config.REPORT_FILE)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_path

def draw_bounding_box(frame, bbox, confidence, is_active, label="Phone"):
    """Draw bounding box with appropriate color and label"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Choose color based on active status
    color = config.BOX_COLOR if is_active else config.INACTIVE_COLOR
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
    
    # Prepare label text
    status = "BEING HELD" if is_active else "NOT HELD"
    label_text = f"{label}: {status} ({confidence:.2f})"
    
    # Calculate text position
    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, config.TEXT_THICKNESS)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1]
    
    # Draw text background
    cv2.rectangle(frame, (text_x, text_y - text_size[1]), 
                  (text_x + text_size[0], text_y + 5), color, -1)
    
    # Draw text
    cv2.putText(frame, label_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, 
                config.TEXT_COLOR, config.TEXT_THICKNESS)
    
    return frame

def draw_tap_to_pay_box(frame, bbox, confidence, is_active, label="TAP TO PAY"):
    """Draw bounding box for tap-to-pay devices with magenta color"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Use magenta color for tap-to-pay devices
    color = config.TAP_TO_PAY_COLOR if is_active else config.INACTIVE_COLOR
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
    
    # Prepare label text
    status = "BEING HELD" if is_active else "NOT HELD"
    label_text = f"{label}: {status} ({confidence:.2f})"
    
    # Calculate text position
    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, config.TEXT_THICKNESS)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1]
    
    # Draw text background
    cv2.rectangle(frame, (text_x, text_y - text_size[1]), 
                  (text_x + text_size[0], text_y + 5), color, -1)
    
    # Draw text
    cv2.putText(frame, label_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, 
                config.TEXT_COLOR, config.TEXT_THICKNESS)
    
    return frame

def get_video_info(video_path):
    """Get video information (fps, frame count, resolution)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
    }