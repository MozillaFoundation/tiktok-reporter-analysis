import os
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
import whisper
from PIL import Image
from transformers import AutoProcessor, IdeficsForVisionText2Text


def set_backend():
    use_mps = torch.backends.mps.is_available()
    print(f"MPS available: {use_mps}")

    # If MPS is available, use it. Otherwise, use CUDA if available, else use CPU
    if use_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


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
    return frames_dataframe


def extract_transcript(video_clip):
    whisper_model = whisper.load_model("base")
    print(whisper_model.device)
    audio = video_clip.audio
    with NamedTemporaryFile(suffix=".wav") as tmpfile:
        audio.write_audiofile(tmpfile.name)
        transcript = whisper_model.transcribe(tmpfile.name)
    return transcript


def multi_modal_analysis(frames, results_path, transcript=None, testing=False):
    with open("./tiktok_reporter_analysis/prompts/idefics_system_prompt.txt", "r") as f:
        SYSTEM_PROMPT = f.readlines()
    SYSTEM_PROMPT[-1] = SYSTEM_PROMPT[-1][:-1]  # Remove EOF newline

    with open("./tiktok_reporter_analysis/prompts/idefics_prompt.txt", "r") as f:
        PROMPT = f.read()[:-1]
    if transcript:
        PROMPT += "\n" + transcript["text"]

    print("Prompt:")
    print(PROMPT)

    frames_to_timestamps = frames.set_index("frame")["timestamp"].to_dict()

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

    prompts = []
    videos = frames["video"].unique()
    for video in videos:
        current_frames = frames.loc[frames["video"] == video, "image"].to_list()
        image1 = current_frames[0]
        image2 = current_frames[1]

        prompts += [
            SYSTEM_PROMPT
            + [
                "\nUser:",
                image1,
                image2,
                PROMPT,
                "<end_of_utterance>",
                "\nAssistant:",
            ],
        ]

    # --batched mode
    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    # --single sample mode
    # inputs = processor(prompts[0], return_tensors="pt").to(device)

    # Generation args
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=1500)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    output_df = pd.DataFrame(
        {
            "video": [video for video in videos],
            "frame1": [frames.loc[frames["video"] == video, "frame"].iloc[0] for video in videos],
            "frame2": [frames.loc[frames["video"] == video, "frame"].iloc[1] for video in videos],
            "description": [generated_text[video].split("\n")[16:][-1].split("Assistant: ")[-1] for video in videos],
        }
    )
    output_df["timestamp1"] = format_ms_timestamp(output_df["frame1"].map(frames_to_timestamps))
    output_df["timestamp2"] = format_ms_timestamp(output_df["frame2"].map(frames_to_timestamps))
    output_df = output_df[["video", "frame1", "timestamp1", "frame2", "timestamp2", "description"]]
    os.makedirs(results_path, exist_ok=True)
    output_df.to_parquet(results_path + "/video_descriptions.parquet")
