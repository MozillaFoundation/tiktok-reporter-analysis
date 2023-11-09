import os

import numpy as np
import pandas as pd
import torch
import whisper
from moviepy.editor import VideoFileClip
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


def extract_frames(frames):
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


def extract_frames_n_audio(video_path, audio_path, results_path, num_frames=2):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(audio_path)

    n_frames_in_video = int(clip.fps * clip.duration)
    frame_timestamps = np.linspace(0, clip.duration, n_frames_in_video)
    selected_frames = extract_frames(frame_timestamps)

    frames = {
        np.where(frame_timestamps == time)[0][0]: Image.fromarray(clip.get_frame(time)) for time in selected_frames
    }

    # Create a pandas dataframe with frame number and timestamp
    df = pd.DataFrame({"frame": list(frames.keys()), "timestamp": selected_frames})
    df["timestamp"] = df["timestamp"] * 1000  # Convert to milliseconds

    # Write the dataframe to a csv file
    df.to_csv(os.path.join(results_path, "frames_n_timestamps.csv"), index=False)

    return frames


def extract_transcript(audio_path):
    whisper_model = whisper.load_model("base")
    print(whisper_model.device)
    transcript = whisper_model.transcribe(audio_path)
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

    frames_to_timestamps = pd.read_csv(os.path.join(results_path, "frames_n_timestamps.csv"), index_col=0)["timestamp"]

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
    for video in frames.keys():
        current_frames = list(frames[video].keys())
        image1 = frames[video][current_frames[0]]
        image2 = frames[video][current_frames[1]]

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
            "video": [video for video in frames.keys()],
            "frame1": [list(frames[video].keys())[0] for video in frames.keys()],
            "frame2": [list(frames[video].keys())[1] for video in frames.keys()],
            "description": [
                generated_text[video].split("\n")[16:][-1].split("Assistant: ")[-1] for video in frames.keys()
            ],
        }
    )
    output_df["timestamp1"] = format_ms_timestamp(output_df["frame1"].map(frames_to_timestamps))
    output_df["timestamp2"] = format_ms_timestamp(output_df["frame2"].map(frames_to_timestamps))
    output_df = output_df[["video", "frame1", "timestamp1", "frame2", "timestamp2", "description"]]
    os.makedirs(results_path, exist_ok=True)
    output_df.to_parquet(results_path + "/video_descriptions.parquet")
