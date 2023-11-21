import logging
import os
import shutil

import cv2

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
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Error: Couldn't open the video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames: {frame_count}")

    count = 0
    frames_n_timestamps = []
    while True:
        ret, frame = cap.read()

        # Break the loop if video is ended
        if not ret:
            break

        # Save the current frame as an image
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        frames_n_timestamps.append({"frame": count, "timestamp": timestamp})
        frame_filename = os.path.join(output_folder, f"frame_{count:04}.jpg")
        cv2.imwrite(frame_filename, frame)

        count += 1
        logger.info(f"Extracted frame {count} of {frame_count}")

    cap.release()
    logger.info("Frames extraction completed.")
    return frames_n_timestamps
