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


def create_prompts_for_idefics(frames, videos, prompt, transcripts=None):
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
                ("Transcript: " + transcripts[(video_path, video)]) if transcripts else "" "<end_of_utterance>",
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
                {"type": "text", "text": prompt.format(transcript=transcript if transcript else "")},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image2}"}},
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
    buf2 = io.BytesIO()
    # video_path_filename = os.path.splitext(os.path.basename(video_path))[0]
    # image1_filename = os.path.join("data/results", f"{video_path_filename}_image1.png")
    # image1.save(image1_filename, format="PNG")
    image1.save(buf1, format="PNG")
    # image2_filename = os.path.join("data/results", f"{video_path_filename}_image2.png")
    # image2.save(image2_filename, format="PNG")
    image2.save(buf2, format="PNG")

    prompt_messages = [
        {
            "role": "user",
            "content": prompt.format(
                transcript=(
                    "Make sure you don't miss these unbelievable astronomical events on June, because on "
                    "June 4th is the first full moon of the month, also known as the Strawberry Moon. And "
                    "on the same day is the best time to view Venus since it will be at its highest point "
                    "above the horizon in the evening sky. On June 7th to 10th, the daytime area-tid meteor "
                    "shower producing its peak rate of meteors. On June 12th to 13th, Venus can be seen in or "
                    "very near the Beehive cluster. A good pair of binoculars should be enough to see this "
                    "rare event. On June 17th you can see a stunning planetary alignment. Saturn, Neptune, "
                    "Jupiter, Mercury and Uranus will line up about an hour before sunrise. And the even "
                    "cooler thing is, you can see this rare event with your naked eye. On June 18th is New "
                    "Moon. That means the moon will not be visible in the night sky. This is also the best "
                    "time of the month to observe faint objects such as galaxies and star clusters, because "
                    "there is no moon light to interfere. And now don't forget to send this video to your "
                    "friends so that they don't miss these beautiful events."
                )
            ),
            "images": [
                "tiktok_reporter_analysis/prompts/7240137209767120155inf_image1.png",
                "tiktok_reporter_analysis/prompts/7240137209767120155inf_image2.png",
            ],
        },
        {
            "role": "assistant",
            "content": "".join(
                "The video appears to be an informative piece about astronomical events scheduled for June. "
                "It mentions various celestial occurrences such as the Strawberry Moon, Venus's position in "
                "relation to a star cluster, meteor showers, and planetary alignments. The content of the "
                "frames is consistent with this description, featuring images related to space and astronomy.\n"
                "category: informative"
            ),
        },
        {
            "role": "user",
            "content": prompt.format(
                transcript=(
                    "Mit seit einer halben Stunde so schlecht und schwindlich, ja keiner an mir was los ist, ich "
                    "hab mich schon Cola geholt, weil ich gehört, davon muss man sich übergeben. Ich meine, jetzt "
                    "sind schon etwas, das heißt, ich hatte das noch, ich bin jetzt eigentlich nie schlecht, nie "
                    "schwindlich. Ich hab keiner an was, was für ein Unruhmann ist schon überlegt. Ich weiß, dass "
                    "das Ding, ich muss ein bisschen leise reden, weil Henrys unten und ich will ihn nicht "
                    "enttäuschen. Und ja, ich hab immer so schaut, auch mal mal an. Ganz mir ist so schlecht. Ich "
                    "hab mir wirklich noch nie gewünscht, dass ich mich beigeben muss, außer jetzt. 5 Minuten. 1. "
                    "Schade, ich hätte mich echt gefreut."
                )
            ),
            "images": [
                "tiktok_reporter_analysis/prompts/7254948333599460635rel_image1.png",
                "tiktok_reporter_analysis/prompts/7254948333599460635rel_image2.png",
            ],
        },
        {
            "role": "assistant",
            "content": "".join(
                "The woman seems to be discussing a pregnancy and relationship related situation.\n"
                "category: relationships",
            ),
        },
        {
            "role": "user",
            "content": prompt.format(transcript=transcript if transcript else ""),
            "images": [buf1.getvalue(), buf2.getvalue()],
        },
    ]

    return prompt_messages


def create_prompt_for_llama(description):
    prompt_messages = [
        {
            "role": "user",
            "content": (
                "Given the following description, please choose one category from this list: "
                "['dance_music', 'comedy_drama', 'entertaiment', 'society', 'cars', "
                "'lifestyle', 'relationships', 'pets_nature', 'sport', 'fashion', 'informative']. "
                'Reply with only the category name from that list. The description is: "{}"'
            ).format(description),
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
    elif model == "lmstudio":
        generated_text = multi_modal_analysis_gpt(
            frames, results_path, PROMPT, transcripts, testing, videos, "http://localhost:1234/v1"
        )
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


def multi_modal_analysis_llava(
    frames,
    results_path,
    PROMPT,
    transcripts=None,
    testing=False,
    videos=None,
):
    results = []
    # input("Press 'c' to continue...")
    for idx, video in enumerate(videos, start=1):
        print(f"Starting to process video number {idx}")
        current_video = video
        current_transcript = transcripts[current_video]
        prompt = create_prompt_for_llava(frames, current_video[0], current_video[1], PROMPT, current_transcript)
        print(f"Creating prompt with transcript length={len(current_transcript['text'])}")
        print(f"Current video={current_video}")
        # print(f"Current transcript is \"{current_transcript}\"")
        response = ollama.chat(model="llava:13b-v1.6", messages=prompt, keep_alive=-1)
        prompt = create_prompt_for_llama(response["message"]["content"])
        response = ollama.chat(model="llama2", messages=prompt, keep_alive=-1)
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
        prompts = create_prompts_for_idefics(frames, current_batch_videos, PROMPT, current_batch_transcripts)
        generated_text += zip(current_batch_videos, generate_batch(prompts, model, processor, device))

    logger.info("Saving results")
    return {video: g.split("\n")[3:][-1].split("Assistant: ")[-1] for video, g in generated_text}


def multi_modal_analysis_gpt(frames, results_path, PROMPT, transcripts=None, testing=False, videos=None, server=None):
    if server:
        client = OpenAI(
            base_url=server,
            api_key="notneeded",
        )
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    logger.info("Using OpenAI API")
    results = []
    for idx, video in enumerate(videos, start=1):
        print(f"Starting to process video number {idx}")
        current_video = video
        current_transcript = transcripts[current_video]
        prompt = create_prompt_for_gpt(frames, current_video[0], current_video[1], PROMPT, current_transcript)
        result = client.chat.completions.create(
            messages=prompt,
            model="gpt-4-vision-preview",
            max_tokens=500,
        )

        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Given the following text please choose whether to classify the video as 'informative' "
                            f"or 'other'. Please output nothing but one of those two words. The text is "
                            f"{result.choices[0].message.content}"
                        ),
                    },
                ],
            }
        ]
        result2 = client.chat.completions.create(
            messages=prompt,
            model="gpt-4-vision-preview",
            max_tokens=500,
        )

        # results.append((video, (result.choices[0].message.content + result2.choices[0].message.content)))
        results.append((video, result2.choices[0].message.content))
    logger.info("Saving results")

    return {video: r for video, r in results}


def multi_modal_from_saved(results_path, testing=False):
    frames, transcripts = load_frames_and_transcripts(results_path)
    multi_modal_analysis(frames, results_path, transcripts, testing=testing)
