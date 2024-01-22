import logging
import os

import pandas as pd
import torch
from transformers import AutoProcessor, IdeficsForVisionText2Text

from tiktok_reporter_analysis.common import (
    format_ms_timestamp,
    load_frames_and_transcripts,
    set_backend,
)

logger = logging.getLogger(__name__)


def create_prompts(frames, videos, system_prompt, prompt, transcripts=None):
    prompts = []
    for video_path, video in videos:
        current_frames = frames.loc[
            (frames["video"] == video) & (frames["video_path"] == video_path), "image"
        ].to_list()
        image1 = current_frames[0]
        image2 = current_frames[1]

        if transcripts:
            CURRENT_PROMPT = (
                prompt
                + " The following line is a audio transcript to give some more context.\n"
                + transcripts[(video_path, video)]["text"]
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
    device = set_backend(no_mps=True)
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
    videos = frames.set_index(["video_path", "video"]).index.unique()
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
            "video_path": [video_path for video_path, _ in videos],
            "video": [video for _, video in videos],
            "frame1": [
                frames.loc[(frames["video"] == video) & (frames["video_path"] == video_path), "frame"].iloc[0]
                for video_path, video in videos
            ],
            "frame2": [
                frames.loc[(frames["video"] == video) & (frames["video_path"] == video_path), "frame"].iloc[1]
                for video_path, video in videos
            ],
            "description": [
                generated_text[video].split("\n")[16:][-1].split("Assistant: ")[-1] for video in range(len(videos))
            ],
            "audio_transcript": [transcripts[(video_path, video)]["text"] for video_path, video in videos],
        }
    )
    output_df["timestamp1"] = format_ms_timestamp(output_df["frame1"].map(frames_to_timestamps))
    output_df["timestamp2"] = format_ms_timestamp(output_df["frame2"].map(frames_to_timestamps))
    output_df = output_df[
        ["video_path", "video", "frame1", "timestamp1", "frame2", "timestamp2", "description", "audio_transcript"]
    ]
    os.makedirs(results_path, exist_ok=True)
    output_df.to_parquet(results_path + "/video_descriptions.parquet")
    logger.info("Results saved")


def multi_modal_from_saved(results_path, testing=False):
    frames, transcripts = load_frames_and_transcripts(results_path)
    multi_modal_analysis(frames, results_path, transcripts, testing=testing)
