import argparse

import whisper
from moviepy.editor import VideoFileClip

from tiktok_reporter_analysis.common import (
    extract_frames,
    extract_transcript,
    multi_modal_analysis,
    set_backend,
)


def classify_reported(video_path, results_path, testing=False):
    video_clip = VideoFileClip(video_path)
    frames_dataframe = extract_frames(video_clip, all_frames=False)
    whisper_model = whisper.load_model("base", device=set_backend())
    print(whisper_model.device)
    transcript = extract_transcript(video_clip, whisper_model)

    print(transcript["text"])

    frames_dataframe["video"] = 0

    multi_modal_analysis(frames_dataframe, results_path, transcripts={0: transcript}, testing=testing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract transcript and frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("audio_path", type=str, help="Path to the audio track")
    parser.add_argument("results_path", type=str, help="Path to results directory")

    args = parser.parse_args()

    classify_reported(args.video_path, args.audio_path)
