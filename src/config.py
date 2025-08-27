"""
Configuration parameters for phone usage detection
"""

# Model paths
PHONE_MODEL_PATH = "../models/best.pt"  # Relative path to model weights

# Detection thresholds
PHONE_CONFIDENCE_THRESHOLD = 0.5
HAND_CONFIDENCE_THRESHOLD = 0.7
PHONE_HAND_DISTANCE_THRESHOLD = 200  # pixels - optimized for reliable hand detection

# Motion analysis parameters
MIN_MOTION_FRAMES = 3  # Reduced from 5 for faster detection
MOTION_THRESHOLD = 5   # Reduced from 10 for more sensitive motion detection
MIN_ACTIVE_FRAMES = 2  # Reduced from 3 for less strict filtering
MAX_INACTIVE_FRAMES = 10

# MediaPipe Hands configuration
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.1  # Very low for maximum hand detection
MIN_TRACKING_CONFIDENCE = 0.1

# Video processing
PRESERVE_AUDIO = True
SAVE_ANNOTATED_VIDEO = True
GENERATE_REPORT = True
OUTPUT_FPS = None  # Same as input video

# Visualization settings
BOX_COLOR = (0, 255, 0)  # Green for active usage
INACTIVE_COLOR = (0, 0, 255)  # Red for inactive
TAP_TO_PAY_COLOR = (255, 0, 255)  # Magenta for tap-to-pay devices
TEXT_COLOR = (255, 255, 255)  # White text
BOX_THICKNESS = 2
TEXT_THICKNESS = 1
TEXT_SCALE = 0.6

# Output settings
OUTPUT_DIR = "output"
REPORT_FILE = "phone_usage_report.json"

# Class mapping for trained model
CLASS_TAP_TO_PAY = 0
CLASS_PHONE = 1
CLASS_NAMES = {0: 'TAP TO PAY DEVICE', 1: 'phone'}

# Timer settings
PHONE_HOLD_TIMER_THRESHOLD = 1.0  # seconds - minimum time to consider as "holding"

# Debug/Demo settings
SHOW_ALL_DETECTIONS = False  # If True, shows all detected devices regardless of hands
DEMO_MODE = False  # If True, treats all detected phones as being held (for demo purposes)