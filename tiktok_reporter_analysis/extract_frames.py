import logging
import os


from tiktok_reporter_analysis.common import extract_frames

logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path, output_folder):
    # Append video filename to output_folder to make it unique
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    unique_output_folder = os.path.join(output_folder, video_filename)

    os.makedirs(unique_output_folder, exist_ok=True)

    # Extract frames using the method from common.py
    frames_dataframe = extract_frames(video_path, unique_output_folder, save_frames=True)

    logger.info("Frames extraction completed.")
    return frames_dataframe
