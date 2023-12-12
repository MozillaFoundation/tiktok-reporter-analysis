import logging
import os

import pandas as pd
import whisper
from moviepy.editor import VideoFileClip

from tiktok_reporter_analysis.analyze_screen_recording import (
    analyze_screen_recording,
    load_checkpoint,
)
from tiktok_reporter_analysis.common import (
    extract_frames,
    extract_transcript,
    get_video_files,
    multi_modal_analysis,
    save_frames_and_transcripts,
    select_frames,
    set_backend,
)

logger = logging.getLogger(__name__)


def classify_videos(video_path, checkpoint_path, results_path, testing=False, multimodal=False, debug=False):
    logger.info(f"Processing screen recordings from {video_path}")
    video_files = get_video_files(video_path)

    logger.info("Loading frame classification and whisper models")
    device = set_backend()
    model = load_checkpoint(checkpoint_path, device)
    whisper_model = whisper.load_model("base", device=device)
    logger.info("Frame classification and whisper models loaded")

    transcripts = {}
    selected_frames_dataframes = []
    for i, video_file in enumerate(video_files):
        logger.info(f"Processing video {i+1}/{len(video_files)}: {video_file}")
        video_clip = VideoFileClip(video_file)
        frames_path = os.path.join(results_path, "frames", os.path.basename(video_file).split(".")[0])
        frames_dataframe = extract_frames(video_clip, frames_path, all_frames=True, debug=debug)

        # analyze screen recordings
        analyze_screen_recording(frames_dataframe, model, device, results_path)

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

        frames_dataframe = pd.merge(frames_dataframe, df[["frame", "video"]], on="frame")

        video_start_end = {video: [video_frames[video][0], video_frames[video][-1]] for video in video_frames.keys()}

        for video in video_frames.keys():
            logger.info(f"Extracting transcript from video {video+1}/{video_counter+1}")
            current_clip = video_clip.subclip(
                video_start_end[video][0] / video_clip.fps, video_start_end[video][1] / video_clip.fps
            )
            transcript = extract_transcript(current_clip, whisper_model)
            transcripts[(video_file, video)] = transcript

        selected_frames = []
        for video in video_frames.keys():
            frames = video_frames[video]
            current_frames = select_frames(frames)
            selected_frames += current_frames

        selected_frames_dataframe = frames_dataframe.loc[selected_frames]
        selected_frames_dataframe["video_file"] = video_file
        selected_frames_dataframes.append(selected_frames_dataframe)

    selected_frames_dataframe = pd.concat(selected_frames_dataframes)
    logger.info("Frames and transcripts extracted")
    if multimodal:
        multi_modal_analysis(selected_frames_dataframe, results_path, transcripts=transcripts, testing=testing)
    else:
        save_frames_and_transcripts(selected_frames_dataframe, transcripts, results_path)


if __name__ == "__main__":
    # Get command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Analyze images in a directory")
    parser.add_argument("frames_dir", type=str, help="Directory of images to analyze")
    parser.add_argument("results_path", type=str, help="Path to results directory")
    args = parser.parse_args()

    classify_videos(args.frames_folder, args.results_path)
