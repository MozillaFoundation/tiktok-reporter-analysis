import argparse

import pandas as pd
import whisper
from moviepy.editor import VideoFileClip

from tiktok_reporter_analysis.common import (
    extract_frames,
    extract_transcript,
    get_video_files,
    multi_modal_analysis,
    set_backend,
)


def classify_reported(video_path, results_path, testing=False):
    video_files = get_video_files(video_path)

    whisper_model = whisper.load_model("base", device=set_backend())

    transcripts = {}
    frames_dataframes = []
    for video_file in video_files:
        print(f"Processing {video_file}")
        video_clip = VideoFileClip(video_file)
        current_frames_dataframe = extract_frames(video_clip, all_frames=False)
        transcript = extract_transcript(video_clip, whisper_model)

        print(f"Transcript: {transcript['text']}")

        current_frames_dataframe["video"] = 0
        current_frames_dataframe["video_file"] = video_file
        frames_dataframes.append(current_frames_dataframe)
        transcripts[(video_file, 0)] = transcript

    frames_dataframe = pd.concat(frames_dataframes)
    multi_modal_analysis(frames_dataframe, results_path, transcripts=transcripts, testing=testing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract transcript and frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("audio_path", type=str, help="Path to the audio track")
    parser.add_argument("results_path", type=str, help="Path to results directory")

    args = parser.parse_args()

    classify_reported(args.video_path, args.audio_path)
