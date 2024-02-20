import logging
import os
import pickle
from tempfile import NamedTemporaryFile

import cv2
import pandas as pd
import torch
from PIL import Image
from moviepy.editor import VideoFileClip

logger = logging.getLogger(__name__)


def set_backend(no_mps=False):
    use_mps = torch.backends.mps.is_available()

    # If MPS is available, use it. Otherwise, use CUDA if available, else use CPU
    if use_mps and not no_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
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


def get_video_paths(video_path):
    if os.path.isdir(video_path):
        video_paths = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".mp4")]
    else:
        video_paths = [video_path]
    return video_paths


def select_frames_int(frames):
    min_frame = min(frames)
    max_frame = max(frames)
    # Calculate one third and two thirds of the way between min and max
    one_third = min_frame + (max_frame - min_frame) // 3
    two_thirds = min_frame + 2 * (max_frame - min_frame) // 3
    # Find the frames that are closest to one third and two thirds of the way between min and max
    current_frames = [min(frames, key=lambda x: abs(x - one_third)), min(frames, key=lambda x: abs(x - two_thirds))]
    return current_frames


def select_frames(frames):
    min_frame = min(frames.frame)
    max_frame = max(frames.frame)
    # Calculate one third and two thirds of the way between min and max
    one_third = min_frame + (max_frame - min_frame) // 3
    two_thirds = min_frame + 2 * (max_frame - min_frame) // 3
    # Find the frames that are closest to one third and two thirds of the way between min and max
    current_frames = [
        min(frames.frame, key=lambda x: abs(x - one_third)),
        min(frames.frame, key=lambda x: abs(x - two_thirds)),
    ]
    return frames[frames.frame.isin(current_frames)]


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


def extract_frames(video_path, frames_path=None):
    pickle_file = os.path.join(frames_path, "frames.pkl") if frames_path else None
    if frames_path and os.path.exists(pickle_file):
        logger.info("Loading frames from pickle")
        with open(pickle_file, 'rb') as f:
            frames_dataframe = pickle.load(f)
    else:
        logger.info("Extracting frames")
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # cv2 can't reliably get FPS or duration, so use moviepy to get duration.
        # We still use cv2 to extract frames as it's faster.
        clip = VideoFileClip(video_path)
        duration = clip.duration

        logger.info(f"video_clip.duration={duration} and frame_count={frame_count}")

        frames_dataframe_rows = []
        logger.info(f"There are {frame_count} frames to process")
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            frames_dataframe_rows.append(
                pd.DataFrame({"frame": [frame_index], "timestamp": [frame_timestamp], "image": [frame_image]})
            )
            if frames_path and False:
                if not os.path.exists(frames_path):
                    os.makedirs(frames_path)
                logger.info("Saving frame to disk")
                frame_image.save(os.path.join(frames_path, f"frame_{frame_index:06d}.png"))
            frame_index = frame_index + 1
        logger.info("Frames extracted")
        frames_dataframe = pd.concat(frames_dataframe_rows, ignore_index=True)
        if pickle_file:
            logger.info("Saving frames to pickle")
            if not os.path.exists(frames_path):
                os.makedirs(frames_path)
            with open(pickle_file, 'wb') as f:
                pickle.dump(frames_dataframe, f)
    return frames_dataframe


def extract_transcript(video_clip, whisper_model):
    audio = video_clip.audio
    with NamedTemporaryFile(suffix=".wav") as tmpfile:
        audio.write_audiofile(tmpfile.name)
        transcript = whisper_model.transcribe(tmpfile.name)
    return transcript["text"]
