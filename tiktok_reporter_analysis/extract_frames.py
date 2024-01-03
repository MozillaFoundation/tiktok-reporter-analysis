import logging
import os
import shutil

from moviepy.editor import VideoFileClip

from tiktok_reporter_analysis.common import extract_frames

logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path, output_folder):
    # Ensure output directory exists and is empty
    if os.path.exists(output_folder):
        confirm = input(f"Output directory {output_folder} already exists. Remove it? (y/n) ")
        if confirm.lower() == "y":
            shutil.rmtree(output_folder)
        else:
            logger.error("Aborting.")
            exit(1)
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = VideoFileClip(video_path)

    # Extract frames using the method from common.py
    frames_dataframe = extract_frames(cap, output_folder, all_frames=True, debug=True)

    logger.info("Frames extraction completed.")
    return frames_dataframe
