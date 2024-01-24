import argparse
import logging
import os

import pandas as pd
import whisper

from moviepy.editor import VideoFileClip

from tiktok_reporter_analysis.common import (
    extract_frames,
    extract_transcript,
    select_frames,
    get_video_paths,
    save_frames_and_transcripts,
    set_backend,
)
from tiktok_reporter_analysis.multimodal import multi_modal_analysis

logger = logging.getLogger(__name__)


def classify_reported(video_path, results_path, prompt_file, model, testing=False, multimodal=False, debug=False):
    logger.info(f"Processing reported videos from {video_path}")
    video_paths = get_video_paths(video_path)

    logger.info("Loading whisper model")
    whisper_model = whisper.load_model("base", device=set_backend(no_mps=True))
    logger.info("Whisper model loaded")

    transcripts = {}
    frames_dataframes = []
    for i, video_path in enumerate(video_paths):
        logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
        frames_path = os.path.join(results_path, "frames", os.path.basename(video_path).split(".")[0])
        current_frames_dataframe = select_frames(extract_frames(video_path, frames_path if debug else None))
        with VideoFileClip(video_path) as video_clip:
            transcript = extract_transcript(video_clip, whisper_model)
        current_frames_dataframe["video"] = 0
        current_frames_dataframe["video_path"] = video_path
        frames_dataframes.append(current_frames_dataframe)
        transcripts[
            (video_path, 0)
        ] = transcript  # For reported videos there is only one video per path, so give it number 0

    frames_dataframe = pd.concat(frames_dataframes)
    logger.info("Frames and transcripts extracted")
    if multimodal:
        multi_modal_analysis(
            frames_dataframe, results_path, prompt_file, model, transcripts=transcripts, testing=testing
        )
    else:
        save_frames_and_transcripts(frames_dataframe, transcripts, results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract transcript and frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("audio_path", type=str, help="Path to the audio track")
    parser.add_argument("results_path", type=str, help="Path to results directory")

    args = parser.parse_args()

    classify_reported(args.video_path, args.audio_path)
