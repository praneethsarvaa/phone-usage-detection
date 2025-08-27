"""
Video Processing Pipeline
Handles video input/output with audio preservation and annotation
"""

import cv2
import os
import time
from moviepy.editor import VideoFileClip, AudioFileClip
import config
import utils
from hand_phone_analyzer import HandPhoneAnalyzer

class VideoProcessor:
    def __init__(self):
        """Initialize the video processor"""
        self.analyzer = HandPhoneAnalyzer()
        self.usage_data = []  # Store usage data for each frame
        
    def process_video(self, input_path, output_path=None):
        """Process video for phone usage detection"""
        print(f"Processing video: {input_path}")
        
        # Get video information
        video_info = utils.get_video_info(input_path)
        print(f"Video info: {video_info['width']}x{video_info['height']}, {video_info['fps']} FPS, {video_info['frame_count']} frames")
        
        # Create output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_annotated.mp4")
        
        # Create output directory
        utils.create_output_directory()
        
        # Open video capture
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = video_info['fps']
        width = video_info['width']
        height = video_info['height']
        total_frames = video_info['frame_count']
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        start_time = time.time()
        
        print("Starting frame processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            phones, tap_to_pay_devices, hands = self.analyzer.process_frame(frame, frame_idx, fps)
            
            # Store usage data
            frame_data = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                "phones_detected": len(phones),
                "tap_to_pay_detected": len(tap_to_pay_devices),
                "hands_detected": len(hands),
                "active_phone_usage": any(phone.get('is_being_held', False) for phone in phones),
                "tap_to_pay_usage": any(device.get('is_being_held', False) for device in tap_to_pay_devices),
                "phone_details": [],
                "tap_to_pay_details": []
            }
            
            # Draw annotations on frame
            annotated_frame = self.annotate_frame(frame, phones, tap_to_pay_devices, hands, frame_idx, fps)
            
            # Store phone details
            for phone in phones:
                frame_data["phone_details"].append({
                    "bbox": phone['bbox'],
                    "confidence": phone['confidence'],
                    "is_being_held": phone.get('is_being_held', False)
                })
            
            # Store tap-to-pay details
            for device in tap_to_pay_devices:
                frame_data["tap_to_pay_details"].append({
                    "bbox": device['bbox'],
                    "confidence": device['confidence'],
                    "is_being_held": device.get('is_being_held', False)
                })
            
            self.usage_data.append(frame_data)
            
            # Write frame
            out.write(annotated_frame)
            
            # Progress update
            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed_time = time.time() - start_time
                fps_processing = frame_idx / elapsed_time
                print(f"Processed {frame_idx}/{total_frames} frames ({fps_processing:.1f} FPS)")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Preserve audio if requested
        if config.PRESERVE_AUDIO:
            print("Preserving audio...")
            self.preserve_audio(input_path, output_path)
        
        # Generate report
        if config.GENERATE_REPORT:
            print("Generating report...")
            report_path = utils.save_report(self.usage_data, input_path, output_path)
            print(f"Report saved to: {report_path}")
        
        # Get final statistics
        stats = self.analyzer.get_usage_statistics()
        print(f"Processing completed!")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Active usage frames: {stats['active_frames']}")
        print(f"Usage percentage: {stats['usage_percentage']:.2f}%")
        print(f"Total phone hold time: {stats['total_phone_hold_time']:.2f}s")
        print(f"Current phone hold time: {stats['current_phone_hold_time']:.2f}s")
        
        return output_path
    
    def annotate_frame(self, frame, phones, tap_to_pay_devices, hands, frame_idx, fps):
        """Annotate frame with bounding boxes and information"""
        annotated_frame = frame.copy()
        
        # Draw phone bounding boxes (only if being held)
        for phone in phones:
            if phone.get('is_being_held', False):  # Only show phones being held
                bbox = phone['bbox']
                confidence = phone['confidence']
                
                annotated_frame = utils.draw_bounding_box(
                    annotated_frame, bbox, confidence, True, config.CLASS_NAMES[config.CLASS_PHONE]
                )
        
        # Draw tap-to-pay device bounding boxes (only if being held)
        for device in tap_to_pay_devices:
            if device.get('is_being_held', False):  # Only show tap-to-pay devices being held
                bbox = device['bbox']
                confidence = device['confidence']
                
                annotated_frame = utils.draw_tap_to_pay_box(
                    annotated_frame, bbox, confidence, True, config.CLASS_NAMES[config.CLASS_TAP_TO_PAY]
                )
        
        # Draw hand landmarks (optional, for debugging)
        if config.MAX_NUM_HANDS > 0:
            for hand in hands:
                landmarks = hand['landmarks']
                for landmark in landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 3, (255, 0, 0), -1)
        
        # Add frame information
        timestamp = frame_idx / fps
        info_text = f"Frame: {frame_idx} | Time: {timestamp:.2f}s | Phones: {len(phones)} | Tap-to-Pay: {len(tap_to_pay_devices)} | Hands: {len(hands)}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add usage status for phones
        phone_being_held = any(phone.get('is_being_held', False) for phone in phones)
        if phone_being_held:
            phone_hold_time = self.analyzer.get_phone_hold_duration()
            status_text = f"ACTIVE PHONE USAGE - Hold Time: {phone_hold_time:.1f}s"
            status_color = (0, 255, 0)
            cv2.putText(annotated_frame, status_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Add status for tap-to-pay devices (no active usage message)
        tap_to_pay_being_held = any(device.get('is_being_held', False) for device in tap_to_pay_devices)
        if tap_to_pay_being_held:
            device_text = "TAP-TO-PAY DEVICE IN USE"
            cv2.putText(annotated_frame, device_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TAP_TO_PAY_COLOR, 2)
        
        return annotated_frame
    
    def preserve_audio(self, input_path, output_path):
        """Preserve original audio in the output video"""
        try:
            # Load original video with audio
            original_video = VideoFileClip(input_path)
            
            # Load processed video (without audio)
            processed_video = VideoFileClip(output_path)
            
            # Combine processed video with original audio
            final_video = processed_video.set_audio(original_video.audio)
            
            # Create temporary output path
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            
            # Write final video with audio
            final_video.write_videofile(temp_output, codec='libx264', audio_codec='aac')
            
            # Replace original output with audio version
            os.replace(temp_output, output_path)
            
            # Cleanup
            original_video.close()
            processed_video.close()
            final_video.close()
            
            print("Audio preserved successfully!")
            
        except Exception as e:
            print(f"Warning: Could not preserve audio: {e}")
            print("Output video will be without audio.")
    
    def get_usage_summary(self):
        """Get summary of phone usage from processed video"""
        if not self.usage_data:
            return None
        
        total_frames = len(self.usage_data)
        active_phone_frames = sum(1 for frame in self.usage_data if frame['active_phone_usage'])
        usage_percentage = (active_phone_frames / total_frames) * 100
        
        # Find phone usage sessions
        usage_sessions = []
        current_session = None
        
        for frame in self.usage_data:
            if frame['active_phone_usage']:
                if current_session is None:
                    current_session = {
                        'start_time': frame['timestamp'],
                        'start_frame': frame['frame_idx']
                    }
            else:
                if current_session is not None:
                    current_session['end_time'] = frame['timestamp']
                    current_session['end_frame'] = frame['frame_idx']
                    current_session['duration'] = current_session['end_time'] - current_session['start_time']
                    usage_sessions.append(current_session)
                    current_session = None
        
        # Handle case where video ends with active usage
        if current_session is not None:
            last_frame = self.usage_data[-1]
            current_session['end_time'] = last_frame['timestamp']
            current_session['end_frame'] = last_frame['frame_idx']
            current_session['duration'] = current_session['end_time'] - current_session['start_time']
            usage_sessions.append(current_session)
        
        return {
            'total_frames': total_frames,
            'active_frames': active_phone_frames,
            'usage_percentage': usage_percentage,
            'usage_sessions': usage_sessions,
            'total_usage_time': sum(session['duration'] for session in usage_sessions),
            'total_phone_hold_time': self.analyzer.get_total_phone_hold_duration()
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.analyzer.cleanup()