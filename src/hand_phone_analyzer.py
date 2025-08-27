"""
Hand-Phone Interaction Analyzer
Combines MediaPipe hand detection with phone detection to determine active usage
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import config
import utils
import time

class HandPhoneAnalyzer:
    def __init__(self):
        """Initialize the hand-phone analyzer with MediaPipe and YOLO models"""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # Initialize YOLO phone detection model
        self.phone_model = YOLO(config.PHONE_MODEL_PATH)
        
        # Initialize tracking variables
        self.phone_positions = []  # Track phone positions over time
        self.active_usage_history = []  # Track active usage over time
        
        # Timer tracking for phone holding
        self.phone_hold_start_time = None
        self.phone_hold_duration = 0.0
        self.current_phone_timer = 0.0
        self.is_phone_being_held = False
        
    def detect_devices(self, frame):
        """Detect phones and tap-to-pay devices in the frame using YOLO"""
        results = self.phone_model(frame, conf=config.PHONE_CONFIDENCE_THRESHOLD)
        phones = []
        tap_to_pay_devices = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    device_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'center': utils.get_phone_center([x1, y1, x2, y2]),
                        'class_id': class_id
                    }
                    
                    if class_id == config.CLASS_PHONE:
                        phones.append(device_info)
                    elif class_id == config.CLASS_TAP_TO_PAY:
                        tap_to_pay_devices.append(device_info)
        
        return phones, tap_to_pay_devices
    
    def detect_hands(self, frame):
        """Detect hands in the frame using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_center = utils.get_hand_center(hand_landmarks)
                hands.append({
                    'landmarks': hand_landmarks,
                    'center': hand_center
                })
        
        return hands
    
    def analyze_phone_hand_interaction(self, phones, hands, frame_shape):
        """Analyze interaction between phones and hands"""
        active_phones = []
        
        for phone in phones:
            phone_center = phone['center']
            is_being_held = False
            
            # Check if any hand is close to the phone (holding/touching)
            for hand in hands:
                hand_center = hand['center']
                if utils.is_phone_hand_close(phone_center, hand_center, frame_shape):
                    is_being_held = True
                    break
            
            phone['is_being_held'] = is_being_held
            phone['is_active'] = is_being_held  # Only phones being held are considered active
            active_phones.append(phone)
        
        return active_phones
    
    def analyze_tap_to_pay_hand_interaction(self, tap_to_pay_devices, hands, frame_shape):
        """Analyze interaction between tap-to-pay devices and hands"""
        active_devices = []
        
        for device in tap_to_pay_devices:
            device_center = device['center']
            is_being_held = False
            
            # Check if any hand is close to the tap-to-pay device
            for hand in hands:
                hand_center = hand['center']
                if utils.is_phone_hand_close(device_center, hand_center, frame_shape):
                    is_being_held = True
                    break
            
            device['is_being_held'] = is_being_held
            device['is_active'] = is_being_held
            active_devices.append(device)
        
        return active_devices
    
    def update_phone_timer(self, phones, fps):
        """Update timer for phone holding duration"""
        phone_currently_held = any(phone.get('is_being_held', False) for phone in phones)
        current_time = time.time()
        
        if phone_currently_held:
            if not self.is_phone_being_held:
                # Phone just started being held
                self.phone_hold_start_time = current_time
                self.is_phone_being_held = True
                self.current_phone_timer = 0.0
            else:
                # Phone continues to be held
                self.current_phone_timer = current_time - self.phone_hold_start_time
        else:
            if self.is_phone_being_held:
                # Phone was just released
                self.phone_hold_duration += self.current_phone_timer
                self.is_phone_being_held = False
                self.current_phone_timer = 0.0
                self.phone_hold_start_time = None
    
    def get_phone_hold_duration(self):
        """Get current phone holding duration"""
        if self.is_phone_being_held and self.phone_hold_start_time:
            return self.current_phone_timer
        return 0.0
    
    def get_total_phone_hold_duration(self):
        """Get total accumulated phone holding duration"""
        total = self.phone_hold_duration
        if self.is_phone_being_held and self.phone_hold_start_time:
            total += self.current_phone_timer
        return total
    
    def update_tracking(self, phones, frame_idx):
        """Update tracking variables for temporal analysis"""
        # Update phone positions
        if phones:
            # Use the first phone's position (assuming single phone scenario)
            self.phone_positions.append(phones[0]['center'])
        else:
            self.phone_positions.append(None)
        
        # Keep only recent history
        max_history = max(config.MIN_MOTION_FRAMES, config.MAX_INACTIVE_FRAMES) * 2
        if len(self.phone_positions) > max_history:
            self.phone_positions = self.phone_positions[-max_history:]
        
        # Update active usage history (only for phones being held)
        is_active = any(phone.get('is_being_held', False) for phone in phones)
        self.active_usage_history.append(is_active)
        
        # Keep only recent history
        if len(self.active_usage_history) > max_history:
            self.active_usage_history = self.active_usage_history[-max_history:]
    
    def apply_temporal_filtering(self, phones):
        """Apply temporal filtering to reduce false positives"""
        if not self.active_usage_history:
            return phones
        
        # Disable aggressive temporal filtering - trust the hand detection
        # Only apply temporal filtering if we have a very long history with no activity
        if len(self.active_usage_history) >= config.MAX_INACTIVE_FRAMES * 3:  # 3x longer history required
            recent_active = self.active_usage_history[-config.MAX_INACTIVE_FRAMES * 2:]  # Check longer period
            recent_activity_rate = sum(recent_active) / len(recent_active)
            
            # Only mark as inactive if NO activity for extended period AND no current hand detection
            if recent_activity_rate == 0.0 and not any(phone.get('is_being_held', False) for phone in phones):
                for phone in phones:
                    phone['is_being_held'] = False
                    phone['is_active'] = False
                    print(f"Temporal filtering: marking phone as inactive due to extended inactivity")
        
        return phones
    
    def process_frame(self, frame, frame_idx, fps=30):
        """Process a single frame for device usage detection"""
        # Detect phones and tap-to-pay devices
        phones, tap_to_pay_devices = self.detect_devices(frame)
        
        # Detect hands
        hands = self.detect_hands(frame)
        
        # Analyze interactions
        phones = self.analyze_phone_hand_interaction(phones, hands, frame.shape)
        tap_to_pay_devices = self.analyze_tap_to_pay_hand_interaction(tap_to_pay_devices, hands, frame.shape)
        
        # Update phone timer
        self.update_phone_timer(phones, fps)
        
        # Update tracking
        self.update_tracking(phones, frame_idx)
        
        # Apply temporal filtering
        phones = self.apply_temporal_filtering(phones)
        
        return phones, tap_to_pay_devices, hands
    
    def get_usage_statistics(self):
        """Get usage statistics from tracking history"""
        if not self.active_usage_history:
            return {
                'total_frames': 0,
                'active_frames': 0,
                'usage_percentage': 0.0,
                'total_phone_hold_time': 0.0,
                'current_phone_hold_time': 0.0
            }
        
        total_frames = len(self.active_usage_history)
        active_frames = sum(self.active_usage_history)
        usage_percentage = (active_frames / total_frames) * 100
        
        return {
            'total_frames': total_frames,
            'active_frames': active_frames,
            'usage_percentage': usage_percentage,
            'total_phone_hold_time': self.get_total_phone_hold_duration(),
            'current_phone_hold_time': self.get_phone_hold_duration()
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.hands.close()