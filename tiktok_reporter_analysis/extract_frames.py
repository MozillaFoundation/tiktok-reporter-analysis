import logging
import os
import shutil


from tiktok_reporter_analysis.common import extract_frames

logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path, output_folder):
    # Append video filename to output_folder to make it unique
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    unique_output_folder = os.path.join(output_folder, video_filename)

    # Ensure unique output directory exists and is empty
    if os.path.exists(unique_output_folder):
        confirm = input(f"Output directory {unique_output_folder} already exists. Remove it? (y/n) ")
        if confirm.lower() == "y":
            shutil.rmtree(unique_output_folder)
        else:
            logger.error("Aborting.")
            exit(1)
    os.makedirs(unique_output_folder, exist_ok=True)

    # Extract frames using the method from common.py
    frames_dataframe = extract_frames(video_path, unique_output_folder, save_frames=True)

    logger.info("Frames extraction completed.")
    return frames_dataframe
