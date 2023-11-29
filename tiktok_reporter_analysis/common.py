import logging
import os
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, IdeficsForVisionText2Text

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


def extract_frames(video_clip, all_frames=False):
    logger.info("Extracting frames")
    n_frames_in_video = int(video_clip.fps * video_clip.duration)
    frame_timestamps = np.linspace(0, video_clip.duration, n_frames_in_video)
    if all_frames:
        selected_frames_timestamps = frame_timestamps
    else:
        selected_frames_timestamps = select_frames(frame_timestamps)

    selected_frames = {
        np.where(frame_timestamps == time)[0][0]: Image.fromarray(video_clip.get_frame(time))
        for time in selected_frames_timestamps
    }
    frames_dataframe = create_frames_dataframe(selected_frames, frame_timestamps)
    logger.info("Frames extracted")
    return frames_dataframe


def extract_transcript(video_clip, whisper_model):
    audio = video_clip.audio
    with NamedTemporaryFile(suffix=".wav") as tmpfile:
        audio.write_audiofile(tmpfile.name)
        transcript = whisper_model.transcribe(tmpfile.name)
    return transcript


def create_prompts(frames, videos, system_prompt, prompt, transcripts=None):
    prompts = []
    for video_file, video in videos:
        current_frames = frames.loc[
            (frames["video"] == video) & (frames["video_file"] == video_file), "image"
        ].to_list()
        image1 = current_frames[0]
        image2 = current_frames[1]

        if transcripts:
            CURRENT_PROMPT = (
                prompt
                + " The following line is a audio transcript to give some more context.\n"
                + transcripts[(video_file, video)]["text"]
            )
        else:
            CURRENT_PROMPT = prompt

        prompts += [
            system_prompt
            + [
                "\nUser:",
                image1,
                image2,
                CURRENT_PROMPT,
                "<end_of_utterance>",
                "\nAssistant:",
            ],
        ]
    return prompts


def generate_batch(prompts, model, processor, device):
    # --batched mode
    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    # --single sample mode
    # inputs = processor(prompts[0], return_tensors="pt").to(device)

    # Generation args
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=1500)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text


def multi_modal_analysis(frames, results_path, transcripts=None, testing=False):
    logger.info("Running multimodal analysis")
    with open("./tiktok_reporter_analysis/prompts/idefics_system_prompt.txt", "r") as f:
        SYSTEM_PROMPT = f.readlines()
    SYSTEM_PROMPT[-1] = SYSTEM_PROMPT[-1][:-1]  # Remove EOF newline

    with open("./tiktok_reporter_analysis/prompts/idefics_prompt.txt", "r") as f:
        PROMPT = f.read()[:-1]

    frames_to_timestamps = frames.set_index("frame")["timestamp"].to_dict()

    logger.info("Loading multimodal model")
    device = set_backend()
    if testing:
        checkpoint = "HuggingFaceM4/tiny-random-idefics"
    else:
        checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    cache_dir = ".cache"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(
        device
    )
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
    logger.info("Multimodal model loaded")

    prompts = []
    videos = frames.set_index(["video_file", "video"]).index.unique()
    batch_size = 8
    n_batches = len(videos) // batch_size + 1
    generated_text = []
    for batch in range(n_batches):
        logger.info(f"Generating for batch {batch + 1} of {n_batches}")
        current_batch_videos = videos[batch * batch_size : (batch + 1) * batch_size]
        current_batch_transcripts = {video_file: transcripts[video_file] for video_file in current_batch_videos}
        prompts = create_prompts(frames, current_batch_videos, SYSTEM_PROMPT, PROMPT, current_batch_transcripts)
        generated_text += generate_batch(prompts, model, processor, device)

    logger.info("Saving results")
    output_df = pd.DataFrame(
        {
            "video_file": [video_file for video_file, _ in videos],
            "video": [video for _, video in videos],
            "frame1": [
                frames.loc[(frames["video"] == video) & (frames["video_file"] == video_file), "frame"].iloc[0]
                for video_file, video in videos
            ],
            "frame2": [
                frames.loc[(frames["video"] == video) & (frames["video_file"] == video_file), "frame"].iloc[1]
                for video_file, video in videos
            ],
            "description": [
                generated_text[video].split("\n")[16:][-1].split("Assistant: ")[-1] for video in range(len(videos))
            ],
            "audio_transcript": [transcripts[(video_file, video)]["text"] for video_file, video in videos],
        }
    )
    output_df["timestamp1"] = format_ms_timestamp(output_df["frame1"].map(frames_to_timestamps))
    output_df["timestamp2"] = format_ms_timestamp(output_df["frame2"].map(frames_to_timestamps))
    output_df = output_df[
        ["video_file", "video", "frame1", "timestamp1", "frame2", "timestamp2", "description", "audio_transcript"]
    ]
    os.makedirs(results_path, exist_ok=True)
    output_df.to_parquet(results_path + "/video_descriptions.parquet")
    logger.info("Results saved")
