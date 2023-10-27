import torch
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


def multi_modal_analysis(frames, results_path, transcript=None):
    with open("./tiktok_reporter_analysis/prompts/idefics_system_prompt.txt", "r") as f:
        SYSTEM_PROMPT = f.readlines()
    SYSTEM_PROMPT[-1] = SYSTEM_PROMPT[-1][:-1]  # Remove EOF newline

    with open("./tiktok_reporter_analysis/prompts/idefics_prompt.txt", "r") as f:
        PROMPT = f.read()[:-1]

    device = set_backend()

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

    responses = ["video,frame1,frame2,description"]
    for video in frames.keys():
        without_system_prompt = generated_text[video].split("\n")[16:]
        generated_response = without_system_prompt[-1].split("Assistant: ")[-1]
        current_frames = list(frames[video].keys())
        row = [f"{video},{current_frames[0]},{current_frames[1]},{generated_response}"]
        print(row)
        responses += row

    responses_str = "\n".join(responses)
    with open(results_path + "/video_descriptions.csv", "w") as f:
        f.write(responses_str)
