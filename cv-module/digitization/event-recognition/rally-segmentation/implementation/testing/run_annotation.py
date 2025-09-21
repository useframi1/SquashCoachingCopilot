#!/usr/bin/env python3
"""
Example script to run the manual annotation pipeline

Usage:
    python testing/run_annotation.py testing/videos/your_video.mp4
    python testing/run_annotation.py testing/videos/your_video.mp4 --window-size 30
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from testing.manual_annotation_pipeline import ManualAnnotationPipeline


def main():
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python run_annotation.py <video_path> [--window-size N]")
        print("\nExample:")
        print("  python testing/run_annotation.py testing/videos/rally_sample.mp4")
        print("  python testing/run_annotation.py testing/videos/rally_sample.mp4 --window-size 30")
        
        # List available videos
        videos_dir = Path("testing/videos")
        if videos_dir.exists():
            video_files = list(videos_dir.glob("*.mp4"))
            if video_files:
                print(f"\nAvailable videos in {videos_dir}:")
                for video in video_files:
                    print(f"  {video}")
            else:
                print(f"\nNo videos found in {videos_dir}")
                print("Please add your test videos to testing/videos/")
        return 1
    
    video_path = sys.argv[1]
    
    # Parse window size if provided
    window_size = 50  # default
    if "--window-size" in sys.argv:
        try:
            idx = sys.argv.index("--window-size")
            window_size = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Error: --window-size requires an integer value")
            return 1
    
    # Validate video file
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    # Check if config exists
    config_path = "pipeline/config.json"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Make sure you're running from the implementation directory")
        return 1
    
    try:
        print(f"Starting annotation pipeline for: {video_path}")
        print(f"Analysis window size: {window_size} frames")
        print("\nControls:")
        print("  s: Mark as START state")
        print("  a: Mark as ACTIVE state") 
        print("  e: Mark as END state")
        print("  SPACE: Pause/Resume")
        print("  ←→: Seek ±1 second")
        print("  ↑↓: Seek ±10 seconds")
        print("  q: Quit and save")
        print("\nPress any key in the video window to continue...")
        
        # Create and run pipeline
        pipeline = ManualAnnotationPipeline(
            video_path=video_path,
            config_path=config_path,
            window_size=window_size
        )
        pipeline.run()
        
        print("Annotation session completed!")
        return 0
        
    except KeyboardInterrupt:
        print("\nAnnotation interrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())