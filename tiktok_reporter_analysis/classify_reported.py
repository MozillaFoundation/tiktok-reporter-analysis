import argparse
import logging
import os

import pandas as pd


from moviepy.editor import VideoFileClip


from tiktok_reporter_analysis.common import (
    extract_frames,
    extract_transcript,
    get_video_paths,
    save_frames_and_transcripts,
)
from tiktok_reporter_analysis.multimodal import multi_modal_analysis

logger = logging.getLogger(__name__)


def classify_reported(
    video_path,
    results_path,
    prompt_file,
    fs_example_file,
    backend,
    model,
    modality_image,
    modality_text,
    modality_video,
    multimodal,
    twopass,
):
    logger.info(f"Processing reported videos from {video_path}")
    video_paths = get_video_paths(video_path)

    transcripts = {}
    frames_dataframes = []
    for i, video_path in enumerate(video_paths):
        logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
        frames_path = os.path.join(results_path, "frames", os.path.basename(video_path).split(".")[0])
        current_frames_dataframe = extract_frames(video_path, frames_path, True)
        with VideoFileClip(video_path) as video_clip:
            transcript_file = os.path.join(
                results_path, "temp", "whisper_cache", os.path.basename(video_path).split(".")[0] + ".txt"
            )
            if os.path.exists(transcript_file):
                with open(transcript_file, "r") as file:
                    transcript = file.read()
            else:
                transcript = extract_transcript(video_clip)
                os.makedirs(os.path.dirname(transcript_file), exist_ok=True)
                with open(transcript_file, "w") as file:
                    file.write(transcript)
        current_frames_dataframe["video"] = 0
        current_frames_dataframe["video_path"] = video_path
        frames_dataframes.append(current_frames_dataframe)
        transcripts[(video_path, 0)] = (
            transcript  # For reported videos there is only one video per path, so give it number 0
        )

    frames_dataframe = pd.concat(frames_dataframes)
    logger.info("Frames and transcripts extracted")
    import time

    if multimodal:
        start_time = time.time()
        multi_modal_analysis(
            frames_dataframe,
            results_path,
            prompt_file,
            fs_example_file,
            backend,
            model,
            transcripts,
            modality_image,
            modality_text,
            modality_video,
            twopass,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open("time.txt", "w") as file:
            file.write(f"Multimodal analysis took {elapsed_time} seconds.")
    else:
        save_frames_and_transcripts(frames_dataframe, transcripts, results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract transcript and frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("audio_path", type=str, help="Path to the audio track")
    parser.add_argument("results_path", type=str, help="Path to results directory")

    args = parser.parse_args()

    classify_reported(args.video_path, args.audio_path)
