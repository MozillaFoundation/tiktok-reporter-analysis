import argparse
from tempfile import NamedTemporaryFile

from tiktok_reporter_analysis.common import (
    extract_frames_n_audio,
    extract_transcript,
    multi_modal_analysis,
)


def classify_reported(video_path, results_path, testing=False):
    with NamedTemporaryFile(suffix=".wav") as tmpfile:
        frames = extract_frames_n_audio(video_path, tmpfile.name, results_path)
        transcript = extract_transcript(tmpfile.name)

    print(transcript["text"])

    multi_modal_analysis({0: frames}, results_path, transcript=transcript, testing=testing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract transcript and frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("audio_path", type=str, help="Path to the audio track")
    parser.add_argument("results_path", type=str, help="Path to results directory")

    args = parser.parse_args()

    classify_reported(args.video_path, args.audio_path)
