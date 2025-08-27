#!/usr/bin/env python3
"""
Phone Usage Detection System
Main entry point for processing videos to detect active phone usage
"""

import argparse
import os
import sys
import time
from video_processor import VideoProcessor
import config

def main():
    """Main function for phone usage detection"""
    parser = argparse.ArgumentParser(
        description="Detect active phone usage in videos using YOLOv8 + MediaPipe Hands"
    )
    
    parser.add_argument(
        "input_video",
        help="Path to input video file (MP4, AVI, MOV)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output video path (default: output/input_annotated.mp4)"
    )
    
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio preservation in output video"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable generation of usage report"
    )
    
    parser.add_argument(
        "--phone-conf",
        type=float,
        default=config.PHONE_CONFIDENCE_THRESHOLD,
        help=f"Phone detection confidence threshold (default: {config.PHONE_CONFIDENCE_THRESHOLD})"
    )
    
    parser.add_argument(
        "--hand-conf",
        type=float,
        default=config.HAND_CONFIDENCE_THRESHOLD,
        help=f"Hand detection confidence threshold (default: {config.HAND_CONFIDENCE_THRESHOLD})"
    )
    
    parser.add_argument(
        "--distance-threshold",
        type=int,
        default=config.PHONE_HAND_DISTANCE_THRESHOLD,
        help=f"Phone-hand distance threshold in pixels (default: {config.PHONE_HAND_DISTANCE_THRESHOLD})"
    )
    
    parser.add_argument(
        "--show-hands",
        action="store_true",
        help="Show hand landmarks in output video (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file '{args.input_video}' does not exist.")
        sys.exit(1)
    
    # Check file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    file_ext = os.path.splitext(args.input_video)[1].lower()
    if file_ext not in valid_extensions:
        print(f"Warning: File extension '{file_ext}' may not be supported.")
        print(f"Supported formats: {', '.join(valid_extensions)}")
    
    # Update configuration based on arguments
    if args.no_audio:
        config.PRESERVE_AUDIO = False
    
    if args.no_report:
        config.GENERATE_REPORT = False
    
    config.PHONE_CONFIDENCE_THRESHOLD = args.phone_conf
    config.HAND_CONFIDENCE_THRESHOLD = args.hand_conf
    config.PHONE_HAND_DISTANCE_THRESHOLD = args.distance_threshold
    
    if args.show_hands:
        config.MAX_NUM_HANDS = 2  # Enable hand visualization
    
    # Print configuration
    print("Phone Usage Detection System")
    print("=" * 40)
    print(f"Input video: {args.input_video}")
    print(f"Phone confidence threshold: {config.PHONE_CONFIDENCE_THRESHOLD}")
    print(f"Hand confidence threshold: {config.HAND_CONFIDENCE_THRESHOLD}")
    print(f"Phone-hand distance threshold: {config.PHONE_HAND_DISTANCE_THRESHOLD} pixels")
    print(f"Preserve audio: {config.PRESERVE_AUDIO}")
    print(f"Generate report: {config.GENERATE_REPORT}")
    print("=" * 40)
    
    # Initialize video processor
    processor = VideoProcessor()
    
    try:
        # Process video
        start_time = time.time()
        output_path = processor.process_video(args.input_video, args.output)
        processing_time = time.time() - start_time
        
        print("\n" + "=" * 40)
        print("PROCESSING COMPLETED")
        print("=" * 40)
        print(f"Output video: {output_path}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Get usage summary
        summary = processor.get_usage_summary()
        if summary:
            print(f"\nUSAGE SUMMARY:")
            print(f"Total frames: {summary['total_frames']}")
            print(f"Active usage frames: {summary['active_frames']}")
            print(f"Usage percentage: {summary['usage_percentage']:.2f}%")
            print(f"Total usage time: {summary['total_usage_time']:.2f} seconds")
            print(f"Number of usage sessions: {len(summary['usage_sessions'])}")
            
            if summary['usage_sessions']:
                print(f"\nUSAGE SESSIONS:")
                for i, session in enumerate(summary['usage_sessions'], 1):
                    print(f"  Session {i}: {session['start_time']:.2f}s - {session['end_time']:.2f}s "
                          f"(Duration: {session['duration']:.2f}s)")
        
        print(f"\nSuccess! Check the output directory for results.")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        processor.cleanup()

if __name__ == "__main__":
    main()






