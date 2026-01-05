"""Convert demo videos to optimized GIFs for GitHub README."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import subprocess
from pathlib import Path

# Configuration
VIDEO_DIR = os.path.join(PROJECT_ROOT, "demo_videos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "demo_videos", "gifs")

# GIF optimization settings
GIF_CONFIGS = {
    "full": {
        "fps": 10,          # Frames per second
        "scale": 640,       # Width in pixels (-1 preserves aspect ratio)
        "max_duration": None,  # Use full video
        "description": "Full video, ~20-30 MB per minute"
    },
    "preview": {
        "fps": 8,
        "scale": 480,
        "max_duration": 30,  # First 30 seconds only
        "description": "30-second preview, ~8-10 MB"
    },
    "compact": {
        "fps": 5,
        "scale": 400,
        "max_duration": 20,  # First 20 seconds
        "description": "20-second compact, ~3-5 MB (recommended)"
    }
}


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL,
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_to_gif(video_path, output_path, fps=10, scale=640, max_duration=None):
    """
    Convert video to optimized GIF using ffmpeg.
    
    Uses two-pass approach for better quality:
    1. Generate palette from video
    2. Use palette to create GIF with dithering
    
    Args:
        video_path: Path to input video
        output_path: Path to output GIF
        fps: Frames per second (lower = smaller file)
        scale: Width in pixels (lower = smaller file)
        max_duration: Max seconds to convert (None = full video)
    """
    palette_path = output_path.replace('.gif', '_palette.png')
    
    # Build ffmpeg commands
    duration_arg = ['-t', str(max_duration)] if max_duration else []
    
    # Pass 1: Generate color palette
    palette_cmd = [
        'ffmpeg', '-y',
        *duration_arg,
        '-i', video_path,
        '-vf', f'fps={fps},scale={scale}:-1:flags=lanczos,palettegen=stats_mode=diff',
        palette_path
    ]
    
    # Pass 2: Create GIF with palette
    gif_cmd = [
        'ffmpeg', '-y',
        *duration_arg,
        '-i', video_path,
        '-i', palette_path,
        '-filter_complex', 
        f'fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
        output_path
    ]
    
    print(f"\nConverting: {os.path.basename(video_path)}")
    print(f"  FPS: {fps}, Scale: {scale}px, Duration: {max_duration or 'full'}")
    
    # Execute commands
    try:
        subprocess.run(palette_cmd, check=True, capture_output=True)
        subprocess.run(gif_cmd, check=True, capture_output=True)
        
        # Clean up palette
        if os.path.exists(palette_path):
            os.remove(palette_path)
        
        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✅ Created: {os.path.basename(output_path)} ({size_mb:.1f} MB)")
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error: {e}")
        if e.stderr:
            print(f"     {e.stderr.decode()}")


def main():
    # Check for ffmpeg
    if not check_ffmpeg():
        print("❌ ffmpeg not found!")
        print("\nInstall ffmpeg:")
        print("  Windows: choco install ffmpeg  OR  download from https://ffmpeg.org/")
        print("  Linux:   sudo apt install ffmpeg")
        print("  macOS:   brew install ffmpeg")
        return
    
    print("✅ ffmpeg found\n")
    
    # Find videos
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(Path(VIDEO_DIR).glob(ext))
    
    if not video_files:
        print(f"❌ No videos found in {VIDEO_DIR}")
        return
    
    print(f"Found {len(video_files)} video(s):")
    for v in video_files:
        print(f"  - {v.name}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Ask user which config to use
    print("\n" + "="*60)
    print("GIF Optimization Presets:")
    print("="*60)
    for name, config in GIF_CONFIGS.items():
        print(f"\n[{name.upper()}]")
        print(f"  {config['description']}")
        print(f"  Settings: {config['fps']}fps, {config['scale']}px, {config['max_duration'] or 'full'} duration")
    
    print("\n" + "="*60)
    choice = input("\nChoose preset [compact/preview/full] (default: compact): ").strip().lower()
    
    if choice not in GIF_CONFIGS:
        choice = 'compact'
    
    config = GIF_CONFIGS[choice]
    print(f"\n✅ Using '{choice.upper()}' preset\n")
    
    # Convert each video
    for video_path in video_files:
        # Output name: "RGB only.mp4" -> "RGB_only_compact.gif"
        base_name = video_path.stem.replace(' ', '_')
        output_name = f"{base_name}_{choice}.gif"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        convert_to_gif(
            str(video_path),
            output_path,
            fps=config['fps'],
            scale=config['scale'],
            max_duration=config['max_duration']
        )
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nGIFs saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Add GIFs to your repository")
    print("2. Update README.md with:")
    print("   ![RGB-only Demo](demo_videos/gifs/RGB_only_compact.gif)")
    print("   ![RGB-Geo Demo](demo_videos/gifs/RGB-Geo_compact.gif)")
    print("   ![RGBD Demo](demo_videos/gifs/RGBD_compact.gif)")


if __name__ == "__main__":
    main()
