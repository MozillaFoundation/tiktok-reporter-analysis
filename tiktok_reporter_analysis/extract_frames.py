import os
import shutil

import cv2
import pandas as pd


def extract_frames_from_video(video_path, output_folder, results_path):
    # Ensure output directory exists and is empty
    if os.path.exists(output_folder):
        confirm = input(f"Output directory {output_folder} already exists. Remove it? (y/n) ")
        if confirm.lower() == "y":
            shutil.rmtree(output_folder)
        else:
            print("Aborting.")
            exit(1)
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")

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
        print(f"Extracted frame {count} of {frame_count}")

    cap.release()
    pd.DataFrame(frames_n_timestamps).to_csv(os.path.join(results_path, "frames_n_timestamps.csv"), index=False)
    print("Frames extraction completed.")


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Extract frames from video")

    # Add the arguments
    parser.add_argument("video_path", type=str, help="The path to the video file")
    parser.add_argument("output_folder", type=str, help="The path to the output folder")
    parser.add_argument("results_path", help="path to the results folder")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function
    extract_frames_from_video(args.video_path, args.output_folder, args.results_path)
