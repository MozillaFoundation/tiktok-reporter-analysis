# NO NEED TO EXTRACT FRAMES AND TRANSCRIPTS IF USING VIDEO MODALITY

import logging
import os
import io
import ollama
import time

import pandas as pd
import json

import base64
import openai

import google.generativeai as genai

from tiktok_reporter_analysis.common import (
    format_ms_timestamp,
    load_frames_and_transcripts,
)


logger = logging.getLogger(__name__)

with open("tiktok_reporter_analysis/prompts/openai_api_key.txt", "r") as key_file:
    OPENAI_API_KEY = key_file.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

with open("tiktok_reporter_analysis/prompts/google_api_key.txt", "r") as key_file:
    GOOGLE_API_KEY = key_file.read().strip()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def image_to_buf(image_path):
    from PIL import Image

    with Image.open(image_path) as image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
    return buf.getvalue()


def create_prompt_for_ollama(
    frames, video_path, video_number, prompt, fs_examples, transcript, modality_image, modality_text
):
    assert modality_image <= 2, "No current support for more than two images"
    assert modality_image > 0, "No current support for zero images"
    current_frames = frames.loc[
        (frames["video"] == video_number) & (frames["video_path"] == video_path), "image"
    ].to_list()
    image1 = current_frames[0]
    image2 = current_frames[1]
    buf1 = io.BytesIO()
    image1.save(buf1, format="PNG")
    buf2 = io.BytesIO()
    image2.save(buf2, format="PNG")
    if fs_examples is None:
        fs_examples = []
    oneimage = modality_image == 1
    prompt_messages = sum(
        [
            [
                {
                    "role": "user",
                    "content": prompt.format(transcript=e["transcript"]),
                    "images": (
                        [image_to_buf(e["image1_path"])]
                        if oneimage
                        else [
                            image_to_buf(e["image1_path"]),
                            image_to_buf(e["image2_path"]),
                        ]
                    ),
                },
                {
                    "role": "assistant",
                    "content": e["response"],
                },
            ]
            for e in fs_examples
        ],
        [],
    ) + [
        {
            "role": "user",
            "content": prompt.format(transcript=transcript if (modality_text and transcript) else ""),
            "images": (
                [buf1.getvalue()]
                if oneimage
                else [
                    buf1.getvalue(),
                    buf2.getvalue(),
                ]
            ),
        },
    ]
    return prompt_messages


def create_prompt_for_openai(
    frames, video_path, video_number, prompt, fs_examples, transcript, modality_image, modality_text
):
    assert modality_image <= 2, "No current support for more than two images"
    assert modality_image > 0, "No current support for zero images"
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
    if fs_examples is None:
        fs_examples = []
    oneimage = modality_image == 1
    prompt_messages = sum(
        [
            [
                {
                    "role": "user",
                    "content": (
                        [
                            {
                                "type": "text",
                                "text": prompt.format(transcript=e["transcript"]),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_to_base64(e['image1_path'])}"},
                            },
                        ]
                        if oneimage
                        else [
                            {
                                "type": "text",
                                "text": prompt.format(transcript=e["transcript"]),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_to_base64(e['image1_path'])}"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_to_base64(e['image2_path'])}"},
                            },
                        ]
                    ),
                },
                {
                    "role": "assistant",
                    "content": e["response"],
                },
            ]
            for e in fs_examples
        ],
        [],
    ) + [
        {
            "role": "user",
            "content": (
                [
                    {
                        "type": "text",
                        "text": prompt.format(transcript=transcript if modality_text and transcript else ""),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image1}"}},
                ]
                if oneimage
                else [
                    {"type": "text", "text": prompt.format(transcript=transcript if transcript else "")},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image2}"}},
                ]
            ),
        },
    ]
    return prompt_messages


def multi_modal_analysis(
    frames,
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
):
    frames_to_timestamps = frames.set_index("frame")["timestamp"].to_dict()
    logger.info("Running multimodal analysis")

    with open(prompt_file, "r") as f:
        prompt = f.read()[:-1]

    fs_examples = None
    if fs_example_file:
        with open(fs_example_file, "r") as f:
            fs_examples = json.load(f)

    videos = frames.set_index(["video_path", "video"]).index.unique()

    if backend == "openai":
        assert not modality_video, "Video modality is not supported by openai backend"
        generated_text = multi_modal_analysis_openai(
            model, frames, prompt, fs_examples, transcripts, videos, modality_image, modality_text, twopass
        )
    elif backend == "ollama":
        assert not modality_video, "Video modality is not supported by ollama backend"
        generated_text = multi_modal_analysis_ollama(
            model, frames, prompt, fs_examples, transcripts, videos, modality_image, modality_text, twopass
        )
    elif backend == "google":
        assert modality_video, "Google backend requires video modality"
        assert not modality_text, "Google backend does not support text modality"
        assert modality_image == 0, "Google backend does not support image modality"
        assert fs_examples is None or fs_examples == "", "Google backend doesn't support few-shot examples"
        assert not twopass, "Google backend doesn't support twopass"
        generated_text = multi_modal_analysis_google(model, frames, prompt, videos)
    else:
        assert False, "Unimplemented backend"
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
            "audio_transcript": [transcripts[(video_path, video)] for video_path, video in videos],
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


def multi_modal_analysis_google(model_name, frames, raw_prompt, videos):
    results = []
    model = genai.GenerativeModel(model_name=model_name)
    for idx, video in enumerate(videos, start=1):
        print(f"Starting to process video number {idx}")
        current_video = video[0]
        print(f"Uploading {current_video}")
        video_file = genai.upload_file(path=current_video) # TODO: batch this
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)
        response = model.generate_content([raw_prompt, video_file], request_options={"timeout": 600})
        results.append((video, response.text))

    return {video: r for video, r in results}


def multi_modal_analysis_ollama(
    model, frames, raw_prompt, fs_examples, transcripts, videos, modality_image, modality_text, twopass
):
    results = []
    for idx, video in enumerate(videos, start=1):
        print(f"Starting to process video number {idx}")
        current_video = video
        current_transcript = transcripts[current_video]
        prompt = create_prompt_for_ollama(
            frames,
            current_video[0],
            current_video[1],
            raw_prompt,
            fs_examples,
            current_transcript,
            modality_image,
            modality_text,
        )
        response = ollama.chat(model=model, messages=prompt, keep_alive=-1)
        result = response["message"]["content"]
        if twopass:
            prompt = [
                {
                    "role": "user",
                    "content": (
                        f"Given the following text please choose whether to classify the video as "
                        f"'informative' or 'other'. Please output nothing but one of those two words. The text "
                        f"is {result}"
                    ),
                }
            ]
            first_result = f"{result}\n\n"
            response = ollama.chat(model="llama2:latest", messages=prompt, keep_alive=-1)
            result = response["message"]["content"]
        else:
            first_result = ""
        results.append((video, f"{first_result}{result}"))
    logger.info("Saving results")

    return {video: r for video, r in results}


def multi_modal_analysis_openai(
    model, frames, raw_prompt, fs_examples, transcripts, videos, modality_image, modality_text, twopass
):
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    logger.info("Using OpenAI API")
    results = []
    for idx, video in enumerate(videos, start=1):
        print(f"Starting to process video number {idx}")
        current_video = video
        current_transcript = transcripts[current_video]
        prompt = create_prompt_for_openai(
            frames,
            current_video[0],
            current_video[1],
            raw_prompt,
            fs_examples,
            current_transcript,
            modality_image,
            modality_text,
        )
        try:
            temp = client.chat.completions.create(
                messages=prompt,
                model=model,
                max_tokens=500,
            )
            result = temp.choices[0].message.content
        except openai.BadRequestError as e:
            logger.error(f"API request error: {e}, Video: {current_video[0]}, Transcript: {current_transcript}")
            result = "model refused"
        if twopass:
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Given the following text please choose whether to classify the video as "
                                f"'informative' or 'other'. Please output nothing but one of those two words. The text "
                                f"is {result}"
                            ),
                        },
                    ],
                }
            ]
            first_result = f"{result}\n\n"
            temp = client.chat.completions.create(
                messages=prompt,
                model=model,
                max_tokens=10,
            )
            result = temp.choices[0].message.content
        else:
            first_result = ""
        results.append((video, f"{first_result}{result}"))
    logger.info("Saving results")

    return {video: r for video, r in results}


def multi_modal_from_saved(results_path):
    frames, transcripts = load_frames_and_transcripts(results_path)
    multi_modal_analysis(frames, results_path, transcripts)
