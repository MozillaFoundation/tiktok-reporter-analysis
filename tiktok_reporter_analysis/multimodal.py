import logging
import os
import io

import pandas as pd
import torch

import base64
from openai import OpenAI
import ollama

from transformers import AutoProcessor, IdeficsForVisionText2Text

from tiktok_reporter_analysis.common import (
    format_ms_timestamp,
    load_frames_and_transcripts,
    set_backend,
)


logger = logging.getLogger(__name__)

with open("tiktok_reporter_analysis/prompts/openai_api_key.txt", "r") as key_file:
    OPENAI_API_KEY = key_file.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def create_prompts_for_llama(frames, videos, prompt, transcripts=None):
    prompts = []
    for video_path, video in videos:
        current_frames = frames.loc[
            (frames["video"] == video) & (frames["video_path"] == video_path), "image"
        ].to_list()
        image1 = current_frames[0]
        image2 = current_frames[1]

        prompts += [
            [
                "\nUser:\n",
                prompt,
                image1,
                image2,
                ("Transcript: " + transcripts[(video_path, video)]["text"]) if transcripts else "" "<end_of_utterance>",
                "\nAssistant:",
            ],
        ]
    return prompts


def create_prompt_for_gpt(frames, video_path, video_number, prompt, transcript=None):
    current_frames = frames.loc[
        (frames["video"] == video_number) & (frames["video_path"] == video_path), "image"
    ].to_list()
    image1 = current_frames[0]
    image2 = current_frames[1]
    buf = io.BytesIO()
    image1.save(buf, format="JPEG")
    encoded_image1 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf = io.BytesIO()
    image2.save(buf, format="JPEG")
    encoded_image2 = base64.b64encode(buf.getvalue()).decode("utf-8")
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image2}"}},
                {"type": "text", "text": "Transcript: " + transcript["text"] if transcript else ""},
            ],
        }
    ]
    return prompt_messages


def create_prompt_for_llava(frames, video_path, video_number, prompt, transcript=None):
    current_frames = frames.loc[
        (frames["video"] == video_number) & (frames["video_path"] == video_path), "image"
    ].to_list()
    image1 = current_frames[0]
    image2 = current_frames[1]
    buf1 = io.BytesIO()
    image1.save(buf1, format="PNG")
    buf2 = io.BytesIO()
    image2.save(buf2, format="PNG")
    prompt_messages = [
        {
            "role": "user",
            "content": prompt + (transcript["text"] if transcript else ""),
            "images": [buf1.read(), buf2.read()],
        },
    ]

    return prompt_messages


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


def multi_modal_analysis(
    frames,
    results_path,
    prompt_file,
    model,
    transcripts=None,
    testing=False,
):
    frames_to_timestamps = frames.set_index("frame")["timestamp"].to_dict()
    logger.info("Running multimodal analysis")

    with open(prompt_file, "r") as f:
        PROMPT = f.read()[:-1]

    videos = frames.set_index(["video_path", "video"]).index.unique()
    if model == "idefics":
        generated_text = multi_modal_analysis_idefics(frames, results_path, PROMPT, transcripts, testing, videos)
    elif model == "gpt":
        generated_text = multi_modal_analysis_gpt(frames, results_path, PROMPT, transcripts, testing, videos)
    elif model == "llava":
        generated_text = multi_modal_analysis_llava(frames, results_path, PROMPT, transcripts, testing, videos)
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
            "description": [generated_text[video] for video in videos],
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


def multi_modal_analysis_llava(
    frames,
    results_path,
    PROMPT,
    transcripts=None,
    testing=False,
    videos=None,
):
    results = []
    for video in videos:
        current_video = video
        current_transcript = transcripts[current_video]
        prompt = create_prompt_for_llava(frames, current_video[0], current_video[1], PROMPT, current_transcript)
        response = ollama.chat(model="llama2", messages=prompt)
        results.append((video, response["message"]["content"]))
    logger.info("Saving results")

    return {video: r for video, r in results}


def multi_modal_analysis_idefics(
    frames,
    results_path,
    PROMPT,
    transcripts=None,
    testing=False,
    videos=None,
):
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
    batch_size = 8
    n_batches = len(videos) // batch_size + 1
    generated_text = []
    for batch in range(n_batches):
        logger.info(f"Generating for batch {batch + 1} of {n_batches}")
        current_batch_videos = videos[batch * batch_size : (batch + 1) * batch_size]
        current_batch_transcripts = {video_file: transcripts[video_file] for video_file in current_batch_videos}
        prompts = create_prompts_for_llama(frames, current_batch_videos, PROMPT, current_batch_transcripts)
        generated_text += zip(current_batch_videos, generate_batch(prompts, model, processor, device))

    logger.info("Saving results")
    return {video: g.split("\n")[3:][-1].split("Assistant: ")[-1] for video, g in generated_text}


def multi_modal_analysis_gpt(
    frames,
    results_path,
    PROMPT,
    transcripts=None,
    testing=False,
    videos=None,
):
    logger.info("Using OpenAI API")
    results = []
    for video in videos:
        current_video = video
        current_transcript = transcripts[current_video]
        prompt = create_prompt_for_gpt(frames, current_video[0], current_video[1], PROMPT, current_transcript)
        result = client.chat.completions.create(
            messages=prompt,
            model="gpt-4-vision-preview",
            max_tokens=2500,
        )
        results.append((video, result.choices[0].message.content))
    logger.info("Saving results")

    return {video: r for video, r in results}


def multi_modal_from_saved(results_path, testing=False):
    frames, transcripts = load_frames_and_transcripts(results_path)
    multi_modal_analysis(frames, results_path, transcripts, testing=testing)
