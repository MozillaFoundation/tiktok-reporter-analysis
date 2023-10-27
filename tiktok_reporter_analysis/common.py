import torch
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


def multi_modal_analysis(selected_frames, frames_folder, results_path):
    SYSTEM_PROMPT = [  # From https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/blob/main/app_dialogue.py
        """The following is a conversation between a highly knowledgeable and intelligent visual AI assistant, called Assistant, and a human user, called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about the content of visual inputs. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.
    The conversation begins:""",
        """\nUser:""",
        "https://i1.pickpik.com/photos/515/906/355/kittens-cats-pet-cute-preview.jpg",
        "Describe this image.<end_of_utterance>",
        """\nAssistant: Five kittens are standing together in the center of the photograph. From the left to right, there is one orange kitten, two kittens white and black stripes, and two brown kittens with an orange head. They are in bright green grass and it looks like they are moving forward.<end_of_utterance>""",
        "\nUser:How about this image?",
        "https://cdn.pixabay.com/photo/2017/09/25/13/12/puppy-2785074_1280.jpg",
        "Can you describe it too?<end_of_utterance>",
        """\nAssistant: A dog is lying on the floor, looking at the camera. It is looking directly at you. The dog has a white body and brown patches on its face and ears. Its eyes are dark. Its nose is black, and it has long, floppy ears, white paws, long fur, big eyes.<end_of_utterance>""",
        "\nUser: What kind of breed is it?<end_of_utterance>",
        """\nAssistant: It looks like a Cavalier King Charles Spaniel. These breeds are known for their gentle, affectionate nature. Their long ears and big, brown eyes give them a sweet, expressive face. Cavaliers are small dogs weighing between 13 and 20 pounds. They have a silky, medium-length coat that comes in various colors, including black and tan, tricolor, red and white, or black.<end_of_utterance>""",
        "\nUser: What can you tell me about this breed of dogs?<end_of_utterance>",
        """\nAssistant: One specific characteristic of the Cavalier King Charles Spaniel is its friendly and affectionate nature. This breed is known for being extremely sociable and forming strong bonds with their owners. They are often described as "velcro dogs" because they love to be close to their human companions, whether it's sitting on their laps, cuddling on the couch, or simply following them around the house.<end_of_utterance>""",
        "\nUser: ghjkhjabnufs<end_of_utterance>",
        """\nAssistant: That doesn’t seem to be a word. Could you ask me another way?<end_of_utterance>""",
        "\nUser: Do you like Cavalier King Charles Spaniel?<end_of_utterance>",
        """\nAssistant: I do not have personal opinions as I’m just a computer program. However, cavaliers are known for being excellent family pets due to their gentle and patient demeanor, making them great with children and other pets. Their affectionate nature also makes them ideal therapy dogs, as they can provide comfort and emotional support to those in need.<end_of_utterance>""",
        "\nUser: How many dogs do you see in this image?",
        "https://i.dailymail.co.uk/i/pix/2011/07/01/article-2010308-0CD22A8300000578-496_634x414.jpg",
        "<end_of_utterance>",
        """\nAssistant: There is no dogs in this image. The picture shows a tennis player jumping to volley the ball.<end_of_utterance>""",
    ]

    device = set_backend()

    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    cache_dir = ".cache"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(
        device
    )
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)

    PROMPT = """These are a two frames from a tiktok video. Write a one paragraph description of the video based on the two frames. Don't talk about the frames separately but infer what is happening in the whole video based on the frames."""
    prompts = []
    for video in selected_frames.keys():
        current_frames = selected_frames[video]
        image1 = Image.open(f"{frames_folder}/frame_{current_frames[0]:04}.jpg")
        image2 = Image.open(f"{frames_folder}/frame_{current_frames[1]:04}.jpg")

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
    for video in selected_frames.keys():
        without_system_prompt = generated_text[video].split("\n")[16:]
        generated_response = without_system_prompt[-1].split("Assistant: ")[-1]
        row = [f"{video},{selected_frames[video][0]},{selected_frames[video][1]},{generated_response}"]
        print(row)
        responses += row

    responses_str = "\n".join(responses)
    with open(results_path + "/video_descriptions.csv", "w") as f:
        f.write(responses_str)
