import argparse
import logging
import os

import pandas as pd
import whisper
from moviepy.editor import VideoFileClip

from tiktok_reporter_analysis.common import (
    extract_frames,
    extract_transcript,
    get_video_files,
    save_frames_and_transcripts,
    set_backend,
)
from tiktok_reporter_analysis.multimodal import multi_modal_analysis

logger = logging.getLogger(__name__)


def classify_reported(video_path, results_path, testing=False, multimodal=False, debug=False):
    logger.info(f"Processing reported videos from {video_path}")
    video_files = get_video_files(video_path)

    logger.info("Loading whisper model")
    whisper_model = whisper.load_model("base", device=set_backend())
    logger.info("Whisper model loaded")

    transcripts = {}
    frames_dataframes = []
    for i, video_file in enumerate(video_files):
        logger.info(f"Processing video {i+1}/{len(video_files)}: {video_file}")
        video_clip = VideoFileClip(video_file)
        frames_path = os.path.join(results_path, "frames", os.path.basename(video_file).split(".")[0])
        current_frames_dataframe = extract_frames(video_clip, frames_path, all_frames=False)
        if debug:
            extract_frames(video_clip, frames_path, all_frames=True, debug=debug)
        transcript = extract_transcript(video_clip, whisper_model)
        current_frames_dataframe["video"] = 0
        current_frames_dataframe["video_file"] = video_file
        frames_dataframes.append(current_frames_dataframe)
        transcripts[(video_file, 0)] = transcript

    frames_dataframe = pd.concat(frames_dataframes)
    logger.info("Frames and transcripts extracted")
    if multimodal:
        multi_modal_analysis(frames_dataframe, results_path, transcripts=transcripts, testing=testing)
    else:
        save_frames_and_transcripts(frames_dataframe, transcripts, results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract transcript and frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("audio_path", type=str, help="Path to the audio track")
    parser.add_argument("results_path", type=str, help="Path to results directory")

    args = parser.parse_args()

    classify_reported(args.video_path, args.audio_path)
