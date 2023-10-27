import pandas as pd
from PIL import Image

from .common import multi_modal_analysis


def classify_videos(frames_folder, results_path):
    # Load the CSV file
    df = pd.read_csv(results_path + "/frame_classification_data.csv")

    # Initialize a state variable to keep track of whether we are in a scrolling state
    scrolling_state = False
    # Create a new column 'video' and initialize it with zeros
    df["video"] = 0
    # Initialize a video counter
    video_counter = 0

    # Iterate over the DataFrame rows
    for i, row in df.iterrows():
        # If the event is 'Scrolling', set the scrolling state to True
        if row["event_name"] == "Scrolling":
            scrolling_state = True
        # If the event is 'TikTok video player' and we are in a scrolling state, increment the video counter
        # and set the scrolling state to False
        elif row["event_name"] == "TikTok video player" and scrolling_state:
            video_counter += 1
            scrolling_state = False
        # Assign the video counter value to the 'video' column
        df.at[i, "video"] = video_counter
    # Create a dictionary where keys are video numbers and values are lists of all frames
    # with a 'TikTok video player' classification
    video_frames = {
        i: list(df[(df["video"] == i) & (df["event_name"] == "TikTok video player")].index)
        for i in range(0, video_counter + 1)
    }

    selected_frames = {}
    for video in video_frames.keys():
        frames = video_frames[video]
        min_frame = min(frames)
        max_frame = max(frames)
        # Calculate one third and two thirds of the way between min and max
        one_third = min_frame + (max_frame - min_frame) // 3
        two_thirds = min_frame + 2 * (max_frame - min_frame) // 3
        # Find the frames that are closest to one third and two thirds of the way between min and max
        current_frames = [min(frames, key=lambda x: abs(x - one_third)), min(frames, key=lambda x: abs(x - two_thirds))]
        selected_frames[video] = {frame: Image.open(frames_folder + f"/frame_{frame}.jpg") for frame in current_frames}

    multi_modal_analysis(selected_frames, results_path)


if __name__ == "__main__":
    # Get command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Analyze images in a directory")
    parser.add_argument("frames_dir", type=str, help="Directory of images to analyze")
    parser.add_argument("results_path", type=str, help="Path to results directory")
    args = parser.parse_args()

    classify_videos(args.frames_folder, args.results_path)
