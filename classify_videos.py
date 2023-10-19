import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, IdeficsForVisionText2Text

# Load the CSV file
df = pd.read_csv("frame_classification_data.csv")

# Initialize a state variable to keep track of whether we are in a scrolling state
scrolling_state = False
# Create a new column 'video' and initialize it with zeros
df["video"] = 0
# Initialize a video counter
video_counter = 0

# Iterate over the DataFrame rows
for i, row in df.iterrows():
    # If the event is 'Scrolling', set the scrolling state to True
    if row["event_name"] == "Scrolling":
        scrolling_state = True
    # If the event is 'TikTok video player' and we are in a scrolling state, increment the video counter
    # and set the scrolling state to False
    elif row["event_name"] == "TikTok video player" and scrolling_state:
        video_counter += 1
        scrolling_state = False
    # Assign the video counter value to the 'video' column
    df.at[i, "video"] = video_counter
# Create a dictionary where keys are video numbers and values are lists of all frames with a 'TikTok video player' classification
video_frames = {
    i: list(df[(df["video"] == i) & (df["event_name"] == "TikTok video player")].index)
    for i in range(0, video_counter + 1)
}

selected_frames = {}
for video in video_frames.keys():
    frames = video_frames[video]
    min_frame = min(frames)
    max_frame = max(frames)
    # Calculate one third and two thirds of the way between min and max
    one_third = min_frame + (max_frame - min_frame) // 3
    two_thirds = min_frame + 2 * (max_frame - min_frame) // 3
    # Find the frames that are closest to one third and two thirds of the way between min and max
    selected_frames[video] = [
        min(frames, key=lambda x: abs(x - one_third)),
        min(frames, key=lambda x: abs(x - two_thirds)),
    ]


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

checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, cache_dir=".cache").to(device)
processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=".cache")


prompts = []
for video in selected_frames.keys():
    PROMPT = """
    These are a few frames from a tiktok video. Write a one paragraph description of the video.
    """

    current_frames = selected_frames[video]
    image1 = Image.open(f"frames/frame_{current_frames[0]}.jpg")
    image2 = Image.open(f"frames/frame_{current_frames[0]}.jpg")

    prompts += [  #
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


responses = ["video,description"]
for video in selected_frames.keys():
    without_system_prompt = generated_text[video].split("\n")[16:]
    generated_response = without_system_prompt[-1].split("Assistant: ")[-1]
    responses += [f"{video},{generated_response}"]

responses_str = "\n".join(responses)
with open("./video_descriptions.csv", "w") as f:
    f.write(responses_str)
