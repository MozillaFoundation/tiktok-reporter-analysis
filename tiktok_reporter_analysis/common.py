import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile

from moviepy.editor import VideoFileClip

import numpy as np
import pandas as pd
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def set_backend():
    # use_mps = torch.backends.mps.is_available()

    # If MPS is available, use it. Otherwise, use CUDA if available, else use CPU
    # if use_mps:
    #    device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device


def save_frames_and_transcripts(frames, transcripts, results_path):
    logger.info("Saving frames and transcripts")
    output_path = os.path.join(results_path, "intermediate")
    os.makedirs(output_path, exist_ok=True)
    frames["image_mode"] = frames["image"].map(lambda x: x.mode)
    frames["image_size"] = frames["image"].map(lambda x: x.size)
    frames["image"] = frames["image"].map(lambda x: x.tobytes())
    frames.to_parquet(os.path.join(output_path, "frames.parquet.gz"), compression="gzip")
    with open(os.path.join(output_path, "transcripts.pickle"), "wb") as f:
        pickle.dump(transcripts, f)
    logger.info("Frames and transcripts saved")


def load_frames_and_transcripts(results_path):
    logger.info("Loading frames and transcripts")
    output_path = os.path.join(results_path, "intermediate")
    frames = pd.read_parquet(os.path.join(output_path, "frames.parquet.gz"))
    frames["image"] = frames[["image_mode", "image_size", "image"]].apply(
        lambda x: Image.frombytes(x["image_mode"], tuple(x["image_size"]), x["image"]), axis=1
    )
    with open(os.path.join(output_path, "transcripts.pickle"), "rb") as f:
        transcripts = pickle.load(f)
    logger.info("Frames and transcripts loaded")
    return frames, transcripts


def get_video_files(video_path):
    if os.path.isdir(video_path):
        video_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".mp4")]
    else:
        video_files = [video_path]
    return video_files


def select_frames(frames):
    min_frame = min(frames)
    max_frame = max(frames)
    # Calculate one third and two thirds of the way between min and max
    one_third = min_frame + (max_frame - min_frame) // 3
    two_thirds = min_frame + 2 * (max_frame - min_frame) // 3
    # Find the frames that are closest to one third and two thirds of the way between min and max
    current_frames = [min(frames, key=lambda x: abs(x - one_third)), min(frames, key=lambda x: abs(x - two_thirds))]
    return current_frames


def format_ms_timestamp(ms_timestamp_series):
    return pd.to_datetime(ms_timestamp_series, unit="ms").dt.strftime("%M:%S.%f")


def create_frames_dataframe(frames, frame_timestamps):
    # assuming selected_frames is a dictionary with frame numbers as keys and Image objects as values
    df = pd.DataFrame.from_dict(frames, orient="index", columns=["image"])

    # add a 'frame' column with the frame numbers
    df["frame"] = df.index

    # reset the index to start from 0
    df = df.reset_index(drop=True)

    # add a 'timestamp' column with the corresponding timestamps
    df["timestamp"] = df["frame"].apply(lambda x: frame_timestamps[x] * 1000)

    # reorder the columns
    df = df[["frame", "timestamp", "image"]]

    return df


def save_frames_to_disk(frames_dataframe, frames_path):
    os.makedirs(frames_path, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        list(
            executor.map(
                lambda x: x[1]["image"].save(os.path.join(frames_path, f"frame_{x[0]}.png")),
                list(frames_dataframe.iterrows()),
            )
        )


def extract_frames(video_clip, frames_path, all_frames=False, debug=False):
    logger.info("Extracting frames")
    n_frames_in_video = int(video_clip.fps * video_clip.duration)
    print(f"video_clip.fps={video_clip.fps} and video_clip.duration={video_clip.duration}")
    frame_timestamps = np.linspace(0, video_clip.duration, n_frames_in_video)
    if all_frames:
        selected_frames_timestamps = frame_timestamps
    else:
        selected_frames_timestamps = select_frames(frame_timestamps)

    frames_dataframe = pd.DataFrame(columns=["frame", "timestamp", "image"])
    print(f"There are {len(selected_frames_timestamps)} frames to process")
    for time in selected_frames_timestamps:
        frame_index = np.where(frame_timestamps == time)[0][0]
        print(f"frame index is {frame_index}")
        frame_image = Image.fromarray(video_clip.get_frame(time))
        frames_dataframe = pd.concat([frames_dataframe, pd.DataFrame({"frame": [frame_index], "timestamp": [time], "image": [frame_image]})], ignore_index=True)
        if debug:
            logger.info("Saving frame to disk")
            frame_image.save(os.path.join(frames_path, f"frame_{frame_index}.png"))
    logger.info("Frames extracted")
    return frames_dataframe


def extract_transcript(video_clip, whisper_model):
    audio = video_clip.audio
    with NamedTemporaryFile(suffix=".wav") as tmpfile:
        audio.write_audiofile(tmpfile.name)
        transcript = whisper_model.transcribe(tmpfile.name)
    return transcript
